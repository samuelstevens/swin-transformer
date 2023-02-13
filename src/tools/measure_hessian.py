"""
Tries to measure the trace of the training loss's hessian

groovy-grape  8192: -10936.734
fuzzy-fig     8192: -2901.677

groovy-grape 16384: -5994.177
fuzzy-fig    16384: 18954.843
"""
import argparse
import logging
import random

import numpy as np
import pyhessian
import torch

from .. import config, data, models, utils

logger = logging.getLogger("measure-hessian")
logger.setLevel(logging.DEBUG)

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def measure_hessian(model, criterion, dataloader, *, n_examples):
    seen = 0
    mean_trace = 0

    for batch, (inputs, targets) in enumerate(dataloader):
        B, C, W, H = inputs.shape

        if len(targets.shape) > 1:
            targets = targets[:, -1]

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        hess = pyhessian.hessian(model, criterion, data=(inputs, targets), cuda=True)
        trace = hess.trace()
        print(
            f"[batch: {batch}] current iter mean: {np.mean(trace):.3f}, running mean: {mean_trace:.3f}, current iter trace: {trace}",
        )

        mean_trace = (mean_trace * seen + np.mean(trace) * B) / (seen + B)

        seen += B
        if seen >= n_examples:
            break

    return mean_trace


def get_config():
    cfg = config._C.clone()
    cfg.defrost()
    cfg.DATA.DATA_PATH = "/local/scratch/cv_datasets/inat21/resize-192"
    cfg.DATA.DATASET = "inat21"
    cfg.DATA.IMG_SIZE = 192
    cfg.MODEL.TYPE = "swinv2"
    cfg.MODEL.NAME = "swinv2_base_window12"
    cfg.MODEL.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINV2.EMBED_DIM = 128
    cfg.MODEL.SWINV2.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.SWINV2.NUM_HEADS = [4, 8, 16, 32]
    cfg.MODEL.SWINV2.WINDOW_SIZE = 12
    cfg.MODEL.HIERARCHICAL_COEFFS = []
    cfg.TRAIN.DEVICE_BATCH_SIZE = 32
    cfg.TEST.SEQUENTIAL = False
    cfg.HIERARCHICAL = False
    cfg.MODE = "eval"
    cfg.freeze()

    return cfg


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = data.build.build_dataset(
        is_train=True, config=config
    )
    config.freeze()

    sampler_val = torch.utils.data.RandomSampler(dataset_train)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_val,
        batch_size=config.TRAIN.DEVICE_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    return dataset_train, dataloader_train


def load_fuzzy_fig(config):
    checkpoint_file = "/local/scratch/stevens.994/hierarchical-vision/pretrained-checkpoints/fuzzy-fig-192-epoch89.pth"

    dataset, dataloader = build_loader(config)
    model = models.build_model(config)
    utils.load_model_checkpoint(checkpoint_file, model, logger)
    assert isinstance(model.head, torch.nn.Linear)
    return model, dataloader


def load_groovy_grape(config):
    checkpoint_file = "/local/scratch/stevens.994/hierarchical-vision/pretrained-checkpoints/groovy-grape-192-epoch89.pth"

    dataset, dataloader = build_loader(config)
    model = models.build_model(config)
    utils.load_model_checkpoint(checkpoint_file, model, logger)
    assert isinstance(model.head, torch.nn.Linear)

    return model, dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fuzzy-fig", "groovy-grape"])
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_config()

    if args.model == "fuzzy-fig":
        model, dataloader = load_fuzzy_fig(cfg)
    elif args.model == "groovy-grape":
        model, dataloader = load_groovy_grape(cfg)
    else:
        raise ValueError(args.model)

    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    n = 32 * 512
    hess = measure_hessian(model, criterion, dataloader, n_examples=n)
    print(f"{args.model} {n}: {hess:.3f}")


if __name__ == "__main__":
    main()
