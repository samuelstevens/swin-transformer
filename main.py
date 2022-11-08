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
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import AverageMeter
import wandb

from config import get_config
from data import build_loader
from hierarchical import (
    FineGrainedCrossEntropyLoss,
    HierarchicalCrossEntropyLoss,
    accuracy,
)
from logger import WandbWriter, create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from utils import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    batch_size_of,
    load_checkpoint,
    load_pretrained,
    reduce_tensor,
    save_checkpoint,
)


def parse_option():
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
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used (deprecated!)",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
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

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    # First load all config options from wandb if we are doing a sweep
    if config.SWEEP.ENABLED:
        config.defrost()
        config.TRAIN.BASE_LR = wandb.config.train_base_lr
        config.TRAIN.BASE_LR = wandb.config.train_base_lr

    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP == 0 and config.MODEL.LABEL_SMOOTHING > 0.0:
        if config.HIERARCHICAL:
            raise NotImplementedError(
                "We don't support hierarhical loss with label smoothing and no mixup."
            )
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        # If we have mixup, smoothing is handled with mixup label transform
        if config.HIERARCHICAL:
            criterion = HierarchicalCrossEntropyLoss(
                coeffs=config.TRAIN.HIERARCHICAL_COEFFS
            ).to(torch.cuda.current_device())
        else:
            criterion = torch.nn.CrossEntropyLoss()

    logger.info("Loss function: %s", criterion)

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger
        )
        acc1, acc5, loss = validate(
            config, data_loader_val, model, config.TRAIN.START_EPOCH - 1
        )
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
        )
        logger.info("Previously reported best accuracy: %.2f", max_accuracy)
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(
            config, data_loader_val, model, config.TRAIN.START_EPOCH - 1
        )
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
        )

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
            loss_scaler,
        )
        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
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

        acc1, acc5, loss = validate(config, data_loader_val, model, epoch)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
        )
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    loss_scaler,
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(),
            create_graph=is_second_order,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step_update(
                    (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
                )
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # We divide by accumulation steps (not sure why) but it makes
        # the logged values look weird. So I multiply by it to fix that.
        loss_meter.update(
            loss.item() * config.TRAIN.ACCUMULATION_STEPS, batch_size_of(targets)
        )

        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            wd = optimizer.param_groups[0]["weight_decay"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"wd {wd:.4f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
            stats = {
                "train/batch_time": batch_time.val,
                "train/batch_loss": loss_meter.val,
                "train/grad_norm": norm_meter.val,
                "train/loss_scale": scaler_meter.val,
                "memory_mb": memory_used,
                "train/learning_rate": config.TRAIN.BASE_LR,
            }
            if lr_scheduler is not None:
                stats["train/learning_rate"] = lr_scheduler.get_update_values(
                    # Copied from line 326
                    (epoch * num_steps + idx)
                    // config.TRAIN.ACCUMULATION_STEPS
                )[0]

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
def validate(config, data_loader, model, epoch):
    if config.HIERARCHICAL:
        criterion = FineGrainedCrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

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
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")
    wandb_writer.log(
        {
            "val/acc1": acc1_meter.avg,
            "val/acc5": acc5_meter.avg,
            "val/loss": loss_meter.avg,
            "epoch": epoch,
        },
    )

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for _ in range(50):
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
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Use this to calculate accumulation steps
    if "WORLD_SIZE" in os.environ:

        def divide_cleanly(a, b):
            assert a % b == 0, f"{a} / {b} has remainder {a % b}"
            return a // b

        n_procs = int(os.environ["WORLD_SIZE"])
        desired_device_batch_size = divide_cleanly(
            config.TRAIN.GLOBAL_BATCH_SIZE, n_procs
        )
        actual_device_batch_size = config.TRAIN.DEVICE_BATCH_SIZE

        if actual_device_batch_size > desired_device_batch_size:
            print(
                f"Decreasing device batch size from {actual_device_batch_size} to {desired_device_batch_size} so your global bath size is {config.TRAIN.GLOBAL_BATCH_SIZE}, not {desired_device_batch_size * n_procs}!"
            )
            config.TRAIN.ACCUMULATION_STEPS = 1
            config.TRAIN.DEVICE_BATCH_SIZE = desired_device_batch_size
        elif desired_device_batch_size == actual_device_batch_size:
            config.TRAIN.ACCUMULATION_STEPS = 1
        else:
            assert desired_device_batch_size > actual_device_batch_size
            config.TRAIN.ACCUMULATION_STEPS = divide_cleanly(
                desired_device_batch_size, actual_device_batch_size
            )
            print(
                f"Using {config.TRAIN.ACCUMULATION_STEPS} accumulation steps so your global batch size is {config.TRAIN.GLOBAL_BATCH_SIZE}, not {actual_device_batch_size * n_procs}!"
            )

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = (
        config.TRAIN.BASE_LR
        * config.TRAIN.DEVICE_BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR
        * config.TRAIN.DEVICE_BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR
        * config.TRAIN.DEVICE_BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = (
            linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        )
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name=f"{config.EXPERIMENT.NAME}",
    )

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if not args.sweep:
        wandb_writer = WandbWriter(rank=dist.get_rank())
        wandb_writer.init(config)
        main(config)
    else:
        sweep_id = wandb.sweep(sweep=config.SWEEP, project="hierarchical-vision")

        wandb_writer = WandbWriter(rank=dist.get_rank())
        wandb_writer.init(config)

        wandb.agent(sweep_id, function=main, count=4)
