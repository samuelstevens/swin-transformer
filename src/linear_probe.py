"""
Script to tune a linear probe over a pre-trained Swin transformer. 

Main differences to main.py:
1. Uses a simpler training loop.
2. Removes code that I think is dead.
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
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
from .utils import (
    auto_resume_helper,
    batch_size_of,
    find_experiments,
    reduce_tensor,
    save_checkpoint,
)

is_ddp = int(os.environ.get("RANK", -1)) != -1

if is_ddp:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    seed_offset = dist.get_rank()
    is_master_process = dist.get_rank() == 0
else:
    # We are not in a distributed process.
    rank = 0
    world_size = 1
    seed_offset = 0
    is_master_process = True


cudnn.benchmark = True


def make_parser():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="path to config file",
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
    parser.add_argument(
        "--mode",
        help="Which mode to use. 'train' is pre-training, 'tune' is fine-tuning on smaller data and 'eval' is evaluation only.",
        choices=["train", "tune", "eval"],
    )
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
    # low-data-regieme; percentage of training data to be use for fine tuning
    parser.add_argument(
        "--low-data",
        type=float,
        help="percentage of training data (.01 to 1) to be used for fine tuning",
    )

    return parser


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def parse_options(parser):
    args, unparsed = parser.parse_known_args()

    return args


def scale_lr(config):
    # Linearly scale the learning rate according to total batch size. May not be optimal.
    scaling_factor = (
        config.TRAIN.DEVICE_BATCH_SIZE
        * dist.get_world_size()
        * config.TRAIN.ACCUMULATION_STEPS
        / 512.0  # Use 512 as default batch size
    )

    config.defrost()
    config.TRAIN.BASE_LR *= scaling_factor
    config.TRAIN.WARMUP_LR *= scaling_factor
    config.TRAIN.MIN_LR *= scaling_factor
    config.freeze()


def fix_batch_size(config):
    if not is_ddp:
        return

    def divide_cleanly(a, b):
        assert a % b == 0, f"{a} / {b} has remainder {a % b}"
        return a // b

    desired_device_batch_size = divide_cleanly(
        config.TRAIN.GLOBAL_BATCH_SIZE, world_size
    )
    actual_device_batch_size = config.TRAIN.DEVICE_BATCH_SIZE

    if actual_device_batch_size > desired_device_batch_size:
        print(
            f"Decreasing device batch size from {actual_device_batch_size} to {desired_device_batch_size} so your global batch size is {config.TRAIN.GLOBAL_BATCH_SIZE}, not {desired_device_batch_size * world_size}!"
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
            f"Using {config.TRAIN.ACCUMULATION_STEPS} accumulation steps so your global batch size is {config.TRAIN.GLOBAL_BATCH_SIZE}, not {actual_device_batch_size * world_size}!"
        )


def main(config):
    # Need to fix some stuff in the config
    config.defrost()

    # Fix batch size based on desired global batch size and maximum device batch size.
    fix_batch_size(config)
    # Linearly scale learning rate.
    # TODO: scale learning rate for Adam with square root factor.
    # scale_lr(config)

    config.freeze()

    wandb_writer.init(config)

    # Logger initialization
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=rank,
        name=config.EXPERIMENT.NAME,
    )
    if is_master_process:
        path = os.path.join(config.OUTPUT, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    dataset_val, dataloader_val = data.build_val_dataloader(config, is_ddp)
    dataset_train, dataloader_train = data.build_train_dataloader(config, is_ddp)
    mixup_fn = data.build_aug_fn(config)

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = models.build_linear_probe(config)

    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    # Just need a single forward pass to initialize the lazy modules before
    # we initialize the optimizers (which need a reference to all parameters).
    with torch.no_grad():
        inputs, targets = next(iter(dataloader_train))
        inputs = inputs.cuda(non_blocking=True)
        model.forward(inputs)

    optimizer = build_optimizer(config, model)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK],
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

    if not config.MODEL.PRETRAINED:
        raise ValueError("not using a pretrained model")

    # Loading from a pretrained checkpoint that might not have trained on this dataset.
    utils.load_pretrained(config, model_without_ddp.backbone, logger)

    acc1, acc5, loss = validate(
        config, dataloader_val, model, config.TRAIN.START_EPOCH - 1, logger
    )
    logger.info(
        "[acc1: %.1f, acc5: %.1f, loss: %.3f, test images: %d, prev acc1: %.1f]",
        acc1,
        acc5,
        loss,
        len(dataset_val),
        max_accuracy,
    )

    if config.MODE == "eval":
        return

    lr_scheduler = build_scheduler(
        config, optimizer, len(dataloader_train) // config.TRAIN.ACCUMULATION_STEPS
    )

    loss_fn = build_loss_criterion(config)

    if checkpoint_file:
        # Load all the training-related stuff.
        utils.load_optimizer_checkpoint(checkpoint_file, optimizer)
        utils.load_lr_scheduler_checkpoint(checkpoint_file, lr_scheduler)
        utils.load_loss_scaler_checkpoint(checkpoint_file, loss_scaler)

    stopper = EarlyStopper(patience=5, min_delta=0.1)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        is_last_epoch = epoch == (config.TRAIN.EPOCHS - 1)
        if is_ddp:
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
        if is_master_process and (epoch % config.SAVE_FREQ == 0 or is_last_epoch):
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

        acc1, acc5, loss = validate(config, dataloader_val, model, epoch, logger)
        logger.info(f"Top 1 acc. on {len(dataset_val)} val. examples: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

        if stopper.early_stop(loss):
            logger.info(f"Early stopping at epoch {epoch}.")
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

        if grad_norm is not None:  # loss_scaler return None if not update
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
    for idx, (images, target) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if is_ddp:
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
                f"Test: [{idx}/{len(dataloader)}]\t"
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


if __name__ == "__main__":
    parser = make_parser()
    args = parse_options(parser)

    if is_ddp:
        # Initialize the distributed process.
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
        torch.distributed.barrier()

    config = get_config(args)
    torch.cuda.set_device(rank)

    # Set seed appropriately.
    seed = config.SEED + seed_offset
    utils.set_seed(seed)

    wandb_writer = WandbWriter(is_master_process)
    main(config)
