# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

from . import data, models, utils
from .config import get_config
from .hierarchical import FineGrainedCrossEntropyLoss, accuracy
from .logger import WandbWriter, create_logger
from .loss_criterion import build_loss_criterion
from .lr_scheduler import build_scheduler
from .optimizer import build_optimizer
from .utils import auto_resume_helper, batch_size_of, reduce_tensor, save_checkpoint

cudnn.benchmark = True


def make_parser():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument(
        "--mode",
        help="Which mode to use. 'train' is pre-training, 'tune' is fine-tuning on smaller data and 'eval' is evaluation only.",
        choices=["train", "tune", "eval"],
    )

    # for acceleration
    parser.add_argument(
        "--fused_window_process",
        action="store_true",
        help="Fused window shift & window partition, similar for reversed part.",
    )
    parser.add_argument(
        "--fused_layernorm", action="store_true", help="Use fused layernorm."
    )
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument(
        "--optim",
        type=str,
        help="overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.",
    )

    return parser


def parse_options(parser):
    args, unparsed = parser.parse_known_args()

    return args


def main(config):
    # Need to fix some stuff in the config
    config.defrost()

    # Fix batch size based on desired global batch size and maximum device batch size.
    utils.fix_batch_size(config)
    utils.scale_lr(config)

    config.freeze()

    # Logger initialization
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=config.DDP.LOCAL_RANK,
        name=config.EXPERIMENT.NAME,
    )
    wandb_writer.init(config)

    if config.DDP.MASTER:
        path = os.path.join(config.OUTPUT, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    dataset_val, dataloader_val = data.build_val_dataloader(config)

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = models.build_model(config)

    model.cuda()
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    optimizer = build_optimizer(config, model)

    if config.DDP.ENABLED:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.DDP.LOCAL_RANK],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    loss_scaler = torch.cuda.amp.GradScaler()

    max_accuracy = 0.0

    # Resuming
    checkpoint_file = auto_resume_helper(config.OUTPUT)
    if checkpoint_file:
        logger.info("Resuming. [path: %s]", checkpoint_file)
        # Resuming training on this dataset.
        checkpoint = utils.load_model_checkpoint(
            checkpoint_file, model_without_ddp, logger
        )
        logger.warn(checkpoint.state_dict_msg)

        max_accuracy = checkpoint.max_accuracy

        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint.start_epoch
        config.freeze()
    else:
        logger.info("No checkpoint found. [path: %s]", config.OUTPUT)

    if config.MODEL.PRETRAINED:
        # Loading from a pretrained checkpoint that might not be used on this dataset.
        utils.load_pretrained(config, model_without_ddp, logger)

    val_metrics = validate(
        config, dataloader_val, model, config.TRAIN.START_EPOCH - 1, logger
    )
    logger.info(
        "[acc1: %.1f, acc5: %.1f, loss: %.3f, test images: %d, prev acc1: %.1f]",
        val_metrics["val/acc1"],
        val_metrics["val/acc5"],
        val_metrics["val/loss"],
        len(dataset_val),
        max_accuracy,
    )

    if config.MODE == "eval":
        return

    if config.MODE == "throughput":
        throughput(dataloader_val, model, logger)
        return

    dataset_train, dataloader_train = data.build_train_dataloader(config)
    mixup_fn = data.build_aug_fn(config)

    loss_fn = build_loss_criterion(config)
    logger.info("Loss function: %s", loss_fn)

    lr_scheduler = build_scheduler(
        config, optimizer, len(dataloader_train) // config.TRAIN.ACCUMULATION_STEPS
    )

    if checkpoint_file:
        # Resuming training on this dataset.
        utils.load_optimizer_checkpoint(checkpoint_file, optimizer)
        utils.load_lr_scheduler_checkpoint(checkpoint_file, lr_scheduler)
        utils.load_loss_scaler_checkpoint(checkpoint_file, loss_scaler)

    stopper = None
    if config.TRAIN.EARLY_STOPPING.PATIENCE > 0:
        stopper = utils.EarlyStopper(
            metric=config.TRAIN.EARLY_STOPPING.METRIC,
            patience=config.TRAIN.EARLY_STOPPING.PATIENCE,
            min_delta=config.TRAIN.EARLY_STOPPING.MIN_DELTA,
            goal=config.TRAIN.EARLY_STOPPING.GOAL,
        )

    logger.info(
        "Start training. [start epoch: %d, final epoch: %d]",
        config.TRAIN.START_EPOCH,
        config.TRAIN.EPOCHS,
    )
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        is_last_epoch = epoch == (config.TRAIN.EPOCHS - 1)
        if config.DDP.ENABLED:
            dataloader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config,
            model,
            loss_fn,
            dataloader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
            loss_scaler,
            logger,
        )
        if cfg.DDP.MASTER and (
            config.SAVE_FREQ > 0 and epoch % config.SAVE_FREQ == 0 or is_last_epoch
        ):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                max_accuracy,
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
            )

        val_metrics = validate(config, dataloader_val, model, epoch, logger)
        logger.info(
            f"Top 1 acc. on {len(dataset_val)} val. examples: %.1f",
            val_metrics["val/acc1"],
        )
        max_accuracy = max(max_accuracy, val_metrics["val/acc1"])
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

        if stopper and stopper.early_stop(val_metrics):
            logger.info(f"Early stopping at epoch {epoch}.")
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                max_accuracy,
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
            )
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    config,
    model,
    criterion,
    dataloader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    loss_scaler,
    logger,
):
    model.train()
    optimizer.zero_grad()

    grad_norm = None

    num_steps = len(dataloader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(dataloader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)

        loss_scaler.scale(loss).backward()
        loss_scaler.unscale_(optimizer)
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD
            )
        loss_scaler.step(optimizer)
        loss_scaler.update()

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )

        loss_meter.update(loss.item(), batch_size_of(targets))

        if grad_norm is not None:
            norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            wd = optimizer.param_groups[0]["weight_decay"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))}\t"
                f"lr {lr:.6f}\t"
                f"wd {wd:.4f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
            stats = {
                "train/batch_time": batch_time.val,
                "train/batch_loss": loss_meter.val,
                "train/grad_norm": norm_meter.val,
                "train/weight_decay": wd,
                "train/learning_rate": lr,
                "memory_mb": memory_used,
            }

            wandb_writer.log(
                {**stats, "step": epoch * num_steps + idx, "epoch": epoch, "batch": idx}
            )

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training took {datetime.timedelta(seconds=int(epoch_time))}"
    )
    wandb_writer.log(
        {"train/epoch_time": epoch_time, "train/loss": loss_meter.avg, "epoch": epoch},
    )


@torch.no_grad()
def validate(config, dataloader, model, epoch, logger):
    # TODO: make this string into an enum
    if config.HIERARCHY.VARIANT == "multitask":
        criterion = FineGrainedCrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if config.DDP.ENABLED:
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Val: [{idx}/{len(dataloader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")
    val_metrics = {
        "val/acc1": acc1_meter.avg,
        "val/acc5": acc5_meter.avg,
        "val/loss": loss_meter.avg,
        "epoch": epoch,
    }
    wandb_writer.log(val_metrics)

    return val_metrics


@torch.no_grad()
def throughput(dataloader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for _ in range(30):
            model(images)
        torch.cuda.synchronize()
        logger.info("throughput averaged with 30 times")
        tic1 = time.time()
        for _ in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        return


if __name__ == "__main__":
    parser = make_parser()
    args = parse_options(parser)

    cfg = get_config(args)

    cfg.defrost()
    cfg.DDP.ENABLED = int(os.environ.get("RANK", -1)) != -1
    if cfg.DDP.ENABLED:
        cfg.DDP.LOCAL_RANK = dist.get_rank()
        cfg.DDP.WORLD_SIZE = dist.get_world_size()
        cfg.DDP.MASTER = cfg.DDP.LOCAL_RANK == 0
    cfg.freeze()

    if cfg.DDP.ENABLED:
        # Initialize the distributed process.
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=cfg.DDP.WORLD_SIZE,
            rank=cfg.DDP.LOCAL_RANK,
        )
        torch.distributed.barrier()
    torch.cuda.set_device(cfg.DDP.LOCAL_RANK)

    # Set seed appropriately.
    seed_offset = cfg.DDP.LOCAL_RANK
    seed = cfg.SEED + seed_offset
    utils.set_seed(seed)

    wandb_writer = WandbWriter(cfg.DDP.MASTER)
    main(cfg)
