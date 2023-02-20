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
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = "part"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

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

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

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

# Swin Transformer MoE parameters
_C.MODEL.SWIN_MOE = CN()
_C.MODEL.SWIN_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_MOE.IN_CHANS = 3
_C.MODEL.SWIN_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_MOE.MLP_RATIO = 4.0
_C.MODEL.SWIN_MOE.QKV_BIAS = True
_C.MODEL.SWIN_MOE.QK_SCALE = None
_C.MODEL.SWIN_MOE.APE = False
_C.MODEL.SWIN_MOE.PATCH_NORM = True
_C.MODEL.SWIN_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_MOE.TOP_VALUE = 1
_C.MODEL.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_MOE.COSINE_ROUTER = False
_C.MODEL.SWIN_MOE.NORMALIZE_GATE = False
_C.MODEL.SWIN_MOE.USE_BPR = True
_C.MODEL.SWIN_MOE.IS_GSHARD_LOSS = False
_C.MODEL.SWIN_MOE.GATE_NOISE = 1.0
_C.MODEL.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.MODEL.SWIN_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.0
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# [SimMIM] Norm target during training
_C.MODEL.SIMMIM = CN()
_C.MODEL.SIMMIM.NORM_TARGET = CN()
_C.MODEL.SIMMIM.NORM_TARGET.ENABLE = False
_C.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = 47

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
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False

# Hierarchical coefficients for loss
_C.TRAIN.HIERARCHICAL_COEFFS = (1,)

# Weighting of the levels of the tree
_C.TRAIN.WEIGHTING = "uniform"

# Co-efficient value for computing weights
_C.TRAIN.ALPHA = 0.1

# Loss to use
_C.TRAIN.LOSS = "fuzzy-fig"

_C.TRAIN.DATA_PERCENTAGE = 1.0

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
# Experiment Settings
# -----------------------------------------------------------------------------
_C.EXPERIMENT = CN()
# The experiment name. This is a human-readable name that is easy to read.
_C.EXPERIMENT.NAME = "default-dragonfruit"
# The wandb id for logging.
# Generate this id with scripts/generate_wandb_id
_C.EXPERIMENT.WANDB_ID = ""

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

# Whether we are doing hierarchical classification
_C.HIERARCHICAL = False

# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

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
_C.MODE = "train"
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f"args.{name}"):
            return True
        return False

    # merge from specific arguments
    if _check_args("batch_size"):
        config.TRAIN.DEVICE_BATCH_SIZE = args.batch_size
    if _check_args("data_path"):
        config.DATA.DATA_PATH = os.path.abspath(args.data_path)
    if _check_args("zip"):
        config.DATA.ZIP_MODE = True
    if _check_args("cache_mode"):
        config.DATA.CACHE_MODE = args.cache_mode
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
    if _check_args("throughput"):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args("enable_amp"):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args("fused_window_process"):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args("fused_layernorm"):
        config.FUSED_LAYERNORM = True
    # Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args("optim"):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # Use os.environ["LOCAL_RANK"] rather than --local_rank
    if "LOCAL_RANK" in os.environ:
        # set local rank for distributed training
        config.LOCAL_RANK = int(os.environ["LOCAL_RANK"])

    assert config.MODE in (
        "train",
        "test",
        "eval",
    ), f"config MODE must be one of 'train', 'test', or 'eval', not '{config.MODE}'"

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.EXPERIMENT.NAME)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config



#################################### Added below sript to incorporate generation of .yaml file #############################################

import tomli
from . import utils
import dataclasses
import copy
from typing_extensions import Literal

from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)
Distribution = Literal["normal", "uniform", "loguniform"]
T = TypeVar("T", bound="Config")

class Config:
    @classmethod
    def from_dict(cls: Type[T], dct: Dict[str, Any]) -> T:
        for field in dataclasses.fields(cls):
            if (
                isinstance(field.type, type)
                and issubclass(field.type, Config)
                and field.name in dct
                and not isinstance(dct[field.name], field.type)
            ):
                if not isinstance(dct[field.name], dict):
                    logger.warn(
                        "Subdict is not a dict! [cls: %s, field name: %s, field type: %s, actual type: %s]",
                        cls,
                        field.name,
                        field.type,
                        type(dct[field.name]),
                    )
                dct[field.name] = field.type.from_dict(dct[field.name])

        return cls(**dct)

    @classmethod
    def get_toml_name(cls) -> str:
        # Because I'm a bad programmer and I do hacky things.
        return cls.__name__[: cls.__name__.lower().find("config")].lower()

    @classmethod
    def from_existing(cls: Type[T], other: Type[T], **overrides) -> T:
        kwargs = {**dataclasses.asdict(other), **overrides}

        return cls(**kwargs)

    @property
    def pretty(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=4)

    def __str__(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def validate_field(self, fname: str, ftype) -> None:
        choices = get_args(ftype)
        if getattr(self, fname) not in choices:
            raise ValueError(f"self.{fname} must be one of {', '.join(choices)}")

@dataclasses.dataclass(frozen=True)
class RandomVecConfig(Config):
    distribution: Optional[Distribution] = None

    # Distribution keyword args.
    dist_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.distribution is not None:
            self.validate_field("distribution", Distribution)
Layer = Union[
    int,
    Literal[
        "sigmoid",
        "tanh",
        "output",
        "cos",
        "sine",
        "layernorm",
        "groupnorm",
        "1/x",
        "nonlinear-wht",
        "dropout",
    ],
]

@dataclasses.dataclass(frozen=True)
class ProjectionConfig(Config):
    layers: List[Layer] = dataclasses.field(default_factory=lambda: ["output"])
    layer_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, dct) -> "ProjectionConfig":
        """
        I reimplement this method because the toml dict will have a string for layers that needs to be evaluated to a real Python list.
        """
        for key in dct:
            if key == "layers" and isinstance(dct[key], str):
                dct[key] = eval(dct[key])

        return cls(**dct)

PromptType = Literal["uuid", "token", "vocab", "chunk-n", "natural-n"]

@dataclasses.dataclass(frozen=True)
class DataConfig(Config):
    file: str
    overwrite_cache: bool = False
    """
    Can be one of 'uuid', 'token', or 'vocab'.
    * uuid: encodes a uuid as the prompt (typically between 20-30 tokens for GPT2).
    * token: adds a new token to the vocabulary for each chunk (<|start0|>, <|start1|>, etc.)
    * vocab: finds an existing token in the vocabulary that's not in any of the examples and uses it as th e prompt.
    * chunk-n: "Chunk 1: ", "Chunk 2: ", ...
    """
    prompt_type: PromptType = "uuid"

    chunk_length: Union[Literal["longest"], int] = "longest"

    def __post_init__(self) -> None:
        if not os.path.exists(self.file):
            raise ValueError(f"{self.file} does not exist!")

        self.validate_field("prompt_type", PromptType)

        if self.chunk_length != "longest":
            assert isinstance(self.chunk_length, int)

    def get_text(self) -> str:
        assert self.file is not None

        with open(self.file, "r") as file:
            return file.read()

@dataclasses.dataclass(frozen=True)
class ModelConfig(Config):
    language_model_name_or_path: str
    intrinsic_dimension: Optional[int] = None

    # Structure-aware intrinsic dimension (SAID)
    # Has no effect when intrinsic_dimension is None.
    intrinsic_dimension_said: bool = False

    # temperature of 1.0 has no effect, lower tend toward greedy sampling
    temperature: float = 1.0

    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: int = 0

    # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    top_p: float = 0.9

    # primarily useful for CTRL model; in that case, use 1.2
    repetition_penalty: float = 1.0

    # optional stop token (ignore text generated after this token)
    stop_token: Optional[str] = None

    # context window size
    context_window: int = 1024

    # dropout probability for fully connected layers in embeddings, encoder, and pooler, embeddings, and attention.
    dropout: float = 0.0

    # dropout probability for the intrinsic dimension layer(s)
    int_dim_dropout: float = 0.0

    # Whether to use pre-trained weights.
    pretrained: bool = True

    random_vector: RandomVecConfig = dataclasses.field(default_factory=RandomVecConfig)

    projection: ProjectionConfig = dataclasses.field(default_factory=ProjectionConfig)

    normalized: bool = True
    scaled: bool = False
    scaling_factor: float = 1

    def __post_init__(self) -> None:
        assert isinstance(self.random_vector, RandomVecConfig), str(
            type(self.random_vector)
        )
        assert isinstance(self.projection, ProjectionConfig), str(type(self.projection))

SeedSource = Literal["trial", "config", "random"]

@dataclasses.dataclass(frozen=True)
class ExperimentConfig(Config):
    model: ModelConfig
    # tokenizer: TokenizerConfig

    ####Below two lines commented by me ##############
    data: DataConfig
    # training: TrainingConfig

    trials: int = 3
    save_weights: bool = True
    seed_source: SeedSource = "trial"
    seed: int = 0

    def __post_init__(self) -> None:
        self.validate_field("seed_source", SeedSource)