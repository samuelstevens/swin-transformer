"""
Given a dataset that's organized into train and val folders, produces a low-data
split, either based on few-shot (maximum number of examples for a given class) or 
percentage (use X% of the total training data, maintaining proportions of the 
training data across classes).
"""

import argparse
import pathlib

import sklearn.model_selection

from . import helpers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Dataset directory containing train/ and val/ directories.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will contain train/ and val/ directories.",
    )
    parser.add_argument(
        "--fraction",
        default=1.0,
        type=float,
        help="Low data fraction. Must be a float between 0 and 1. Cannot be used with --few-shot",
    )
    parser.add_argument(
        "--few-shot",
        default=0,
        type=int,
        help="Number of examples for each class. Cannot be used with --fraction.",
    )

    return parser.parse_args()


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


def load_data(input_dir: pathlib.Path):
    """
    Assume input_dir contains a train/ and val/ directory.
    Each of these directories contains a list of class directories.
    Each train/<CLASS>/ directory has a list of image files.

    Returns image_paths_train, classes_train, image_paths_val, classes_val
    """
    image_paths_train, classes_train = [], []
    for class_path in (input_dir / "train").iterdir():
        for image_path in class_path.iterdir():
            image_paths_train.append(image_path)
            classes_train.append(class_path.name)

    image_paths_val, classes_val = [], []
    for class_path in (input_dir / "val").iterdir():
        for image_path in class_path.iterdir():
            image_paths_val.append(image_path)
            classes_val.append(class_path.name)

    return image_paths_train, classes_train, image_paths_val, classes_val


def main():
    args = parse_args()

    output_dir = pathlib.Path(args.output)
    logger = helpers.create_logger("low-data-split", output_dir)

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

    # How much of the original data to use.
    low_data_fraction = args.fraction
    few_shot_count = args.few_shot

    assert (
        few_shot_count == 0 or low_data_fraction == 1.0
    ), "You cannot specify both --few-shot and --fraction!"

    if low_data_fraction < 1.0:
        x_train, y_train = stratified_low_data_split(
            x_train, y_train, low_data_fraction
        )
        logger.info("Created low-data split.")
    elif few_shot_count > 0:
        # Assume we are in the few-shot regime.
        raise NotImplementedError("Haven't implemented few-shot splits yet.")
    else:
        raise ValueError("You should specifiy --few-shot or --fraction")

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

    helpers.save_data(x_train, y_train, output_dir / "train")
    helpers.save_data(x_val, y_val, output_dir / "val")
    logger.info("Done. [train: %d, val: %d]", len(y_train), len(y_val))


if __name__ == "__main__":
    main()
