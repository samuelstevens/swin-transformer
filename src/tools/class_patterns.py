"""
Given a trained model and a validation dataloader, do some error analysis on the predictions.

Specifically, find:

1. The highest-accuracy classes.
2. Those classes' names and example images.
3. The lowest-accuracy clases.
4. Those classes' names and example iamges.
5. The predicted classes for the lowest-accuracy clases
6. Those classes' names and example images.

Then package it all up in a nice little report of some kind - maybe make it easy to use this module in jupyter notebooks, so the majority of the output is images rather than code.
"""

import argparse
import logging
import math
import os
import pathlib

import PIL
import torch
from tqdm.auto import tqdm

from .. import config, data, models, utils

logger = logging.getLogger("measure-hessian")
logger.setLevel(logging.DEBUG)

# Global variables/hyperparameters

# Whether to use GPU
device = "cuda" if torch.cuda.is_available else "cpu"
# Root dir for where to store images
root_dir = pathlib.Path("/local/scratch/stevens.994/hierarchical-vision/error-analysis")

# Checkpoints for pretrained groovy and fuzzy fig models.
groovy_grape_checkpoint_file = "/local/scratch/stevens.994/hierarchical-vision/pretrained-checkpoints/groovy-grape-192-epoch89.pth"
fuzzy_fig_checkpoint_file = "/local/scratch/stevens.994/hierarchical-vision/pretrained-checkpoints/fuzzy-fig-192-epoch89.pth"

# Datapath and dataset
data_path = "/local/scratch/cv_datasets/inat21/resize-192"
dataset = "inat21"

# Device batch size (no need to do gradients, so
# can be much higher than training batch size).
batch_size = 512


def get_images(dataset, class_index):
    cls_dir = pathlib.Path(dataset.root, dataset.classes[class_index])

    return [PIL.Image.open(cls_dir / filename) for filename in os.listdir(cls_dir)]


def join_images(images, *, n_wide=None, n_tall=1):
    """
    Assumes all images are the same height and width.
    """

    if not images:
        return None

    if n_wide is not None and n_tall is not None:
        assert (
            len(images) == n_wide * n_tall
        ), "If you specify both n_wide and n_tall, they must fit evenly."

    if n_wide is None:
        n_wide = math.ceil(len(images) / n_tall)
    width = images[0].width * n_wide

    if n_tall is None:
        n_tall = math.ceil(len(images) / n_wide)
    height = images[0].height * n_tall

    composite = PIL.Image.new("RGB", (width, height))
    x, y = 0, 0

    for i, image in enumerate(images):
        composite.paste(image, (x, y))

        if (i + 1) % n_wide == 0:
            y += image.width
            x = 0
        else:
            x += image.width

    return composite


@torch.no_grad()
def get_predictions(model, dataloader):
    predictions, labels = [], []
    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        _, preds = outputs.topk(1, dim=1, largest=True, sorted=True)
        preds = preds.squeeze()

        predictions.append(preds)
        labels.append(targets)

    return torch.concat(predictions, dim=0), torch.concat(labels, dim=0)


def class_accuracy(predictions, labels, n_classes):
    correct_preds = torch.zeros(n_classes, dtype=torch.long, device=device)
    incorrect_preds = torch.zeros(n_classes, dtype=torch.long, device=device)

    correct_preds += torch.bincount(
        labels, weights=(predictions == labels), minlength=n_classes
    ).long()
    incorrect_preds += torch.bincount(
        labels, weights=(predictions != labels), minlength=n_classes
    ).long()

    return correct_preds / (correct_preds + incorrect_preds)


def get_config():
    cfg = config._C.clone()
    cfg.defrost()
    cfg.DATA.DATA_PATH = data_path
    cfg.DATA.DATASET = dataset
    cfg.DATA.IMG_SIZE = 192
    cfg.MODEL.TYPE = "swinv2"
    cfg.MODEL.NAME = "swinv2_base_window12"
    cfg.MODEL.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINV2.EMBED_DIM = 128
    cfg.MODEL.SWINV2.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.SWINV2.NUM_HEADS = [4, 8, 16, 32]
    cfg.MODEL.SWINV2.WINDOW_SIZE = 12
    cfg.MODEL.HIERARCHICAL_COEFFS = []
    cfg.TRAIN.DEVICE_BATCH_SIZE = batch_size
    cfg.TEST.SEQUENTIAL = True
    cfg.HIERARCHICAL = False
    cfg.MODE = "eval"
    cfg.freeze()

    return cfg


def build_loader(config):
    config.defrost()
    dataset_val, config.MODEL.NUM_CLASSES = data.build.build_dataset(
        is_train=False, config=config
    )
    config.freeze()

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.TRAIN.DEVICE_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    return dataset_val, dataloader_val


def load_fuzzy_fig(config):
    dataset, dataloader = build_loader(config)
    model = models.build_model(config)
    utils.load_model_checkpoint(fuzzy_fig_checkpoint_file, model, logger)
    assert isinstance(model.head, torch.nn.Linear)
    return model, dataset, dataloader


def load_groovy_grape(config):
    dataset, dataloader = build_loader(config)
    model = models.build_model(config)
    utils.load_model_checkpoint(groovy_grape_checkpoint_file, model, logger)
    assert isinstance(
        model.head, torch.nn.Linear
    ), "Should only load a species-level head for groovy-grape"

    return model, dataset, dataloader


def understand_predictions(dataset, predictions, labels, class_index, save_dir):
    save_dir = save_dir / dataset.classes[class_index]
    save_dir.mkdir(exist_ok=True, parents=True)

    input_images = get_images(dataset, class_index)
    join_images(get_images(dataset, class_index)).save(
        save_dir / "all-validation-images.png"
    )
    for i, pred in enumerate(predictions[labels == class_index].tolist()):
        predicted_class_img = join_images(
            get_images(dataset, pred), n_wide=5, n_tall=None
        )
        input_img = input_images[i]

        input_img.save(save_dir / f"validation-{i}.png")
        predicted_class_img.save(
            save_dir / f"validation-{i}-predicted-{dataset.classes[pred]}-examples.png"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fuzzy-fig", "groovy-grape"], required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_config()
    if args.model == "fuzzy-fig":
        model, dataset, dataloader = load_fuzzy_fig(cfg)
    elif args.model == "groovy-grape":
        model, dataset, dataloader = load_groovy_grape(cfg)
    else:
        raise ValueError(args.model)
    model = model.to(device)

    predictions, labels = get_predictions(model, dataloader)
    acc = class_accuracy(predictions, labels, 10_000)

    save_dir = root_dir / args.model

    _, lowest_classes = acc.topk(10, largest=False, sorted=True)
    for c in lowest_classes:
        print(
            f"Model has {acc[c].item():.3f} top 1 validation accuracy on class {dataset.classes[c]}"
        )
        understand_predictions(dataset, predictions, labels, c, save_dir)


if __name__ == "__main__":
    main()
