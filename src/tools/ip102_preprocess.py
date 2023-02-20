"""
Turns the raw IP102 dataset into the same format expected by the training pipeline.
"""

import argparse
import pathlib
import shutil

from .. import concurrency
from . import helpers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="IP102 input directory. Contains files like classes.txt and the ip102_v1.1/ directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will contain a train/ and val/ directory.",
    )

    return parser.parse_args()


def load_data(input_dir: pathlib.Path):
    class_names = {}
    # There are some images with label 0 which have no class name.
    # When I manually inspect these classes, there is no pattern to them.
    class_names["0"] = "0-unknown"
    with open(input_dir / "classes.txt") as fd:
        for line in fd:
            class_num, *_ = line.split()
            class_name = "-".join(line.split())
            class_names[class_num] = class_name

    train_paths, train_classes = [], []
    with open(input_dir / "ip102_v1.1" / "train.txt") as fd:
        for line in fd:
            image_path, class_num = line.split()
            train_paths.append(input_dir / "ip102_v1.1" / "images" / image_path)
            train_classes.append(class_names[class_num])

    val_paths, val_classes = [], []
    with open(input_dir / "ip102_v1.1" / "val.txt") as fd:
        for line in fd:
            image_path, class_num = line.split()
            val_paths.append(input_dir / "ip102_v1.1" / "images" / image_path)
            val_classes.append(class_names[class_num])

    return train_paths, train_classes, val_paths, val_classes


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
    x_train, y_train, x_val, y_val = load_data(input_dir)

    dist_train = helpers.ClassDistribution(y_train)
    dist_val = helpers.ClassDistribution(y_val)
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
