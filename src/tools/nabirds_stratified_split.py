"""
Makes a stratified, possibly low-data split of NA birds using sklearn's 
model_selection.train_test_split() method. 
"""
import argparse
import collections
import pathlib
import shutil
import statistics

import sklearn.model_selection

from .. import concurrency
from . import helpers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="NA birds input directory. Contains files like README, images.txt and train_test_split.txt",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will contain a train/ and val/ directory.",
    )
    parser.add_argument(
        "--low-data",
        default=1.0,
        type=float,
        help="Low data fraction. Must be a float between 0 and 1.",
    )
    parser.add_argument(
        "--train-val",
        default=0.8,
        type=float,
        help="Fraction of data to use for training. Must be a float between 0 and 1.",
    )

    return parser.parse_args()


def load_train_data(input_dir: pathlib.Path):
    """
    Returns paths to the images (x) and their classes (y) as a tuple of lists.
    """

    image_path_lookup = {}
    with open(input_dir / "images.txt") as fd:
        for line in fd:
            image_id, image_path = line.split()
            image_path_lookup[image_id] = image_path

    train_images = set()
    with open(input_dir / "train_test_split.txt") as fd:
        for line in fd:
            image_id, is_training = line.split()
            if is_training != "1":
                continue
            train_images.add(image_path_lookup[image_id])

    train_images = sorted(train_images)
    x = [input_dir / "images" / image_path for image_path in train_images]

    class_name_lookup = {}
    with open(input_dir / "classes.txt") as fd:
        for line in fd:
            class_num, *class_name_words = line.split()
            class_num = class_num.rjust(4, "0")
            class_name_lookup[class_num] = " ".join(class_name_words)

    y = []
    for image_path in train_images:
        class_num, _ = image_path.split("/")
        filesystem_safe_class_name = (
            class_name_lookup[class_num]
            .replace("'", "")
            .replace(",", "")
            .replace(" ", "-")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "-")
        )
        y.append(f"{class_num}_{filesystem_safe_class_name}")

    return x, y


def stratified_low_data_split(x, y, low_data_fraction: float):
    assert (
        0 < low_data_fraction < 1
    ), f"Must be a fraction between 0 and 1, not {low_data_fraction}!"

    # We discard the test split because that's the "rest" of the data.
    # We deliberately only want a subset of the original data.
    x, _, y, _ = sklearn.model_selection.train_test_split(
        x, y, train_size=low_data_fraction, random_state=42, stratify=y
    )

    return x, y


def stratified_train_val_split(x, y, train_fraction: float):
    assert (
        0 < train_fraction < 1
    ), f"Must be a fraction between 0 and 1, not {train_fraction}!"

    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x, y, train_size=train_fraction, random_state=42, stratify=y
    )

    return x_train, x_val, y_train, y_val


class ClassDistribution:
    def __init__(self, seq):
        self.counts = collections.Counter(seq)

    def min(self):
        return self.counts.most_common()[-1]

    def max(self):
        return self.counts.most_common(1)[0]

    def mean(self):
        return statistics.mean(self.counts.values())


def save_data(x, y, output_dir: pathlib.Path):
    """
    Copies the images from x into the output_dir. 
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    def output_path_of(example):
        image_path = x[i]
        cls = y[i]
        *_, filename = image_path.parts
        return output_dir / cls / filename

    try:
        pool = concurrency.BoundedExecutor()
        for i, path in enumerate(x):
            pool.submit(output_path_of(i).parent.mkdir, parents=True, exist_ok=True)
        pool.finish(desc="Making directories")

        for i, path in enumerate(x):
            pool.submit(shutil.copy2, str(path), output_path_of(i))
        pool.finish(desc="Copying data")
    finally:
        pool.shutdown()


def main():
    args = parse_args()

    output_dir = pathlib.Path(args.output)
    logger = helpers.create_logger("nabirds-split", output_dir)

    input_dir = pathlib.Path(args.input)
    x, y = load_train_data(input_dir)

    dist = ClassDistribution(y)
    logger.info(
        "Class distribution: [min: %s, mean: %.2f, max: %s]",
        dist.min(),
        dist.mean(),
        dist.max(),
    )

    # How much of the original data to use.
    low_data_fraction = args.low_data
    if low_data_fraction < 1:
        x, y = stratified_low_data_split(x, y, low_data_fraction)
        dist = ClassDistribution(y)
        logger.info(
            "Class distribution after low data split: [min: %s, mean: %.2f, max: %s]",
            dist.min(),
            dist.mean(),
            dist.max(),
        )

    train_val_fraction = args.train_val
    x_train, x_val, y_train, y_val = stratified_train_val_split(
        x, y, train_val_fraction
    )
    dist_train = ClassDistribution(y_train)
    dist_val = ClassDistribution(y_val)
    logger.info(
        "Train class distribution: [min: %s, mean: %.2f, max: %s]",
        dist_train.min(),
        dist_train.mean(),
        dist_train.max(),
    )
    logger.info(
        "Val class distribution: [min: %s, mean: %.2f, max: %s]",
        dist_val.min(),
        dist_val.mean(),
        dist_val.max(),
    )

    save_data(x_train, y_train, output_dir / "train")
    save_data(x_val, y_val, output_dir / "val")
    logger.info("Done. [train: %d, val: %d]", len(y_train), len(y_val))


if __name__ == "__main__":
    main()
