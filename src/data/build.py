# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from timm.data import Mixup, create_transform
from torch.utils.data import Subset
from torchvision import datasets, transforms

from .cached_image_folder import CachedImageFolder
from .constants import data_mean_std
from .hierarchical import HierarchicalImageFolder, HierarchicalMixup
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except ImportError:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_val, config.MODEL.NUM_CLASSES = build_dataset(is_train=False, config=config)
    config.freeze()
    print(f"Rank {config.LOCAL_RANK}/{dist.get_rank()} built val dataset.")

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.TRAIN.DEVICE_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # Make these guys into lists so they have a __len__
    dataset_train = []
    dataloader_train = []

    mixup_fn = None

    if config.MODE in ("train", "tune"):
        dataset_train, _ = build_dataset(is_train=True, config=config)
        print(f"Rank {config.LOCAL_RANK}/{dist.get_rank()} built train dataset.")

        # Check if we are overfitting some subset of the training data for debugging
        if config.TRAIN.OVERFIT_BATCHES > 0:
            n_examples = config.TRAIN.OVERFIT_BATCHES * config.TRAIN.DEVICE_BATCH_SIZE
            indices = random.sample(range(len(dataset_train)), n_examples)
            dataset_train = Subset(dataset_train, indices)
        # Check if training is for low data regieme; select subset of data (script added)
        if config.TRAIN.DATA_PERCENTAGE < 1:
            n_examples = config.TRAIN.DATA_PERCENTAGE * len(dataset_train) 
            indices = random.sample(range(len(dataset_train)), int(n_examples))
            dataset_train = Subset(dataset_train, indices)

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
            indices = np.arange(
                dist.get_rank(), len(dataset_train), dist.get_world_size()
            )
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=config.TRAIN.DEVICE_BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )

        # setup mixup / cutmix
        mixup_fn = None
        mixup_active = (
            config.AUG.MIXUP > 0
            or config.AUG.CUTMIX > 0.0
            or config.AUG.CUTMIX_MINMAX is not None
        )
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=config.AUG.MIXUP,
                cutmix_alpha=config.AUG.CUTMIX,
                cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB,
                switch_prob=config.AUG.MIXUP_SWITCH_PROB,
                mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.MODEL.LABEL_SMOOTHING,
                num_classes=config.MODEL.NUM_CLASSES,
            )
            if config.HIERARCHICAL:
                mixup_fn = HierarchicalMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

    return dataset_train, dataset_val, dataloader_train, dataloader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == "imagenet":
        prefix = "train" if is_train else "val"
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(
                config.DATA.DATA_PATH,
                ann_file,
                prefix,
                transform,
                cache_mode=config.DATA.CACHE_MODE if is_train else "part",
            )
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == "imagenet22K":
        prefix = "ILSVRC2011fall_whole"
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841

    elif config.DATA.DATASET in ("inat21", "inat19"):
        if config.DATA.ZIP_MODE:
            raise NotImplementedError(
                f"We do not support zipped {config.DATA.DATASET}."
            )

        prefix = "train" if is_train else "val"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        if config.HIERARCHICAL:
            dataset = HierarchicalImageFolder(root, transform=transform)
            nb_classes = dataset.num_classes
        else:
            dataset = datasets.ImageFolder(root, transform=transform)
            nb_classes = len(dataset.classes)
    elif config.DATA.DATASET == "tiger-beetle":
        if config.DATA.ZIP_MODE:
            raise NotImplementedError("We do not support zipped tiger-beetle")
        if config.HIERARCHICAL:
            raise NotImplementedError("We do not support hierarchical tiger-beetle")

        prefix = "train" if is_train else "val"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 9
    elif config.DATA.DATASET == "nabird":
        if config.DATA.ZIP_MODE:
            raise NotImplementedError("We do not support zipped nabird")
        if config.HIERARCHICAL:
            raise NotImplementedError("We do not support hierarchical nabird")

        prefix = "train" if is_train else "val"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 555
    elif config.DATA.DATASET == "ip102":
        if config.DATA.ZIP_MODE:
            raise NotImplementedError("We do not support zipped nabird")
        if config.HIERARCHICAL:
            raise NotImplementedError("We do not support hierarchical nabird")

        prefix = "train" if is_train else "val"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 102
    elif config.DATA.DATASET == "stanford_dogs":
        if config.DATA.ZIP_MODE:
            raise NotImplementedError("We do not support zipped nabird")
        if config.HIERARCHICAL:
            raise NotImplementedError("We do not support hierarchical nabird")

        prefix = "train" if is_train else "val"
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 120
    else:
        raise NotImplementedError(f"We do not support dataset '{config.DATA.DATASET}'.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER
            if config.AUG.COLOR_JITTER > 0
            else None,
            auto_augment=config.AUG.AUTO_AUGMENT
            if config.AUG.AUTO_AUGMENT != "none"
            else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4
            )
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(
                    size, interpolation=_pil_interp(config.DATA.INTERPOLATION)
                ),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION),
                )
            )

    if config.DATA.DATA_PATH in data_mean_std:
        mean, std = data_mean_std[config.DATA.DATA_PATH]
    elif config.DATA.DATASET in data_mean_std:
        mean, std = data_mean_std[config.DATA.DATASET]
    else:
        raise RuntimeError(
            f"Can't find mean/std for {config.DATA.DATASET} at {config.DATA.DATA_PATH}. Please add it to data/constants.py (try using python -m data.inat normalize for iNat)."
        )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
