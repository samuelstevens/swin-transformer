# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ""
# Dataset name
_C.DATA.DATASET = "imagenet"
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = "bicubic"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "swin"
# Model name
_C.MODEL.NAME = "swin_tiny_patch4_window7_224"
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight. Could be
# overwritten by command line argument
_C.MODEL.PRETRAINED = ""
# Path to file that maps the pretraining task classes to the current task classes.
# Empty means no such mapping exists.
# src/data/map22kto1k.txt is the map for imagenet22k to imagenet1k.
_C.MODEL.LINEAR_HEAD_MAP_FILE = ""
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.0
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Linear Probe parameters
_C.MODEL.LINEAR_PROBE = CN()
# Use the last layer
_C.MODEL.LINEAR_PROBE.LAYERS = [-1]


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.TRAIN.DEVICE_BATCH_SIZE = 128
# Global batch size = DEVICE_BATCH_SIZE * N_PROCS * ACCUMULATION_STEPS
_C.TRAIN.GLOBAL_BATCH_SIZE = 1024
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
# If WARMUP_LR_FRACTION_OF_BASE_LR > 0, then use BASE_LR * WARMUP_LR_FRACTION_OF_BASE_LR
_C.TRAIN.WARMUP_LR_FRACTION_OF_BASE_LR = 0.0
_C.TRAIN.MIN_LR = 5e-6
# If MIN_LR_FRACTION_OF_BASE_LR > 0, then use BASE_LR * MIN_LR_FRACTION_OF_BASE_LR
_C.TRAIN.MIN_LR_FRACTION_OF_BASE_LR = 0.0
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

_C.TRAIN.EARLY_STOPPING = CN()
# Which metric to track.
_C.TRAIN.EARLY_STOPPING.METRIC = "val/acc1"
# Whether you want to maximize or minimize.
_C.TRAIN.EARLY_STOPPING.GOAL = "max"
# Number of epochs to allow a worsening metric.
# If 0, then do not do any early stopping.
_C.TRAIN.EARLY_STOPPING.PATIENCE = 0
# Min change to count an incoming epoch as "worse"
_C.TRAIN.EARLY_STOPPING.MIN_DELTA = 1.0e-5

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


# [Debugging] How many batches of the training data to overfit.
_C.TRAIN.OVERFIT_BATCHES = 0

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = "pixel"
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = "batch"

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Hierarchical Settings
# -----------------------------------------------------------------------------
_C.HIERARCHY = CN()
# Variant can be one of "multitask", "hxe" or empty ("") for disabled.
_C.HIERARCHY.VARIANT = ""
# Hierarchical coefficients for loss
_C.HIERARCHY.MULTITASK_COEFFS = (1,)
# Weights of the levels of the tree. Can be "uniform" or "exponential"
_C.HIERARCHY.HXE_TREE_WEIGHTS = "uniform"
# Factor for exponential weighting
_C.HIERARCHY.HXE_ALPHA = 0.1


# -----------------------------------------------------------------------------
# Experiment Settings
# -----------------------------------------------------------------------------
_C.EXPERIMENT = CN()
# The experiment name. This is a human-readable name that is easy to read.
_C.EXPERIMENT.NAME = "default-dragonfruit"
# The wandb id for logging.
# Generate this id with scripts/generate_wandb_id
_C.EXPERIMENT.WANDB_ID = ""
# Goal for the experiment
_C.EXPERIMENT.GOAL = ""
# Which version of the experiment you're on (for organizational purposes)
_C.EXPERIMENT.VERSION = 1
# Any tags you know you want to include (for organizational purposes)
_C.EXPERIMENT.TAGS = []

# -----------------------------------------------------------------------------
# Distributed Training Settings
# -----------------------------------------------------------------------------
_C.DDP = CN()
# Assume we are not in distributed data parallel setting
_C.DDP.ENABLED = False
# Assume we are always the master process
_C.DDP.MASTER = True
# Assume there is only one process
_C.DDP.N_PROCS = 1
# Assume local rank is 0
_C.DDP.LOCAL_RANK = 0
# Assume world size is 1
_C.DDP.WORLD_SIZE = 1

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------


# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ""
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# One of "train", "tune" or "eval".
# "train" means pretraining, on large-scale data. This includes wandb logging, slack alerts, easy resuming, but no wandb sweeps.
# "tune" means finetuning, typically on smaller-scale data. This means there is wandb logging and sweeps, but no wandb slack alerts and does not support easily resuming runs.
# "eval" is evaluation only. Nothing is re-initialized, and results are not logged to wandb.
# "throughput" means measuring throughput using Microsoft's original code.
_C.MODE = "train"
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False


def update_config_from_file(config, cfg_file):
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    config.defrost()
    # Set defaults using BASE option.
    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if not cfg:
            continue

        update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))

    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    update_config_from_file(config, args.cfg)

    config.defrost()

    def _check_args(name):
        if hasattr(args, name) and eval(f"args.{name}"):
            return True
        return False

    # merge from specific arguments
    if _check_args("batch_size"):
        config.TRAIN.DEVICE_BATCH_SIZE = args.batch_size
    if _check_args("data_path"):
        config.DATA.DATA_PATH = os.path.abspath(args.data_path)
    if _check_args("pretrained"):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args("use_checkpoint"):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args("disable_amp"):
        config.AMP_ENABLE = False
    if _check_args("output"):
        config.OUTPUT = args.output
    if _check_args("mode"):
        config.MODE = args.mode

    # for acceleration
    if _check_args("fused_window_process"):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args("fused_layernorm"):
        config.FUSED_LAYERNORM = True
    # Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args("optim"):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    assert config.MODE in (
        "train",
        "throughput",
        "eval",
    ), f"config MODE must be one of 'train', 'throughput', or 'eval', not '{config.MODE}'"

    # output folder
    config.OUTPUT = os.path.join(
        config.OUTPUT, *sorted(config.EXPERIMENT.TAGS), config.EXPERIMENT.NAME
    )

    config.freeze()


def get_default_config():
    return _C.clone()


def get_config(args):
    """
    Get a yacs CfgNode object with default values.
    """
    config = get_default_config()
    update_config(config, args)

    return config
