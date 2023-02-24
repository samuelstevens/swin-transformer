"""
Makes a stratified, split of NA birds using sklearn's model_selection.train_test_split() 
method. 
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
        help="NA birds input directory. Contains files like README, images.txt and train_test_split.txt",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will contain a train/ and val/ directory.",
    )
    parser.add_argument(
        "--train-val",
        default=0.8,
        type=float,
        help="Fraction of data to use for training. Must be a float between 0 and 1.",
    )

    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Whether to include the hiearchy in the class names.",
    )
    parser.add_argument(
        "--DEV-ignore-perching-birds",
        action="store_true",
        help="Whether to ignore the 'perching-birds' hierarchy level.",
    )

    return parser.parse_args()


def load_train_data(
    input_dir: pathlib.Path, *, hierarchical=False, DEV_ignore_perching_birds=False
):
    """
    Returns paths to the images (x) and their classes (y) as a tuple of lists.

    If hierarchical is true, use the hierarchy in hierarchy.txt to make longer,
    hierarchical classnames (each tier is separated by an underscore)
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

    def clean_class_name(name):
        return (
            name.replace("'", "")
            .replace(",", "")
            .replace(" ", "-")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "-")
        )

    def clean_class_num(num):
        return num.rjust(4, "0")

    class_name_lookup = {}
    with open(input_dir / "classes.txt") as fd:
        for line in fd:
            class_num, *class_name_words = line.split()
            class_num = clean_class_num(class_num)
            class_name_lookup[class_num] = clean_class_name(" ".join(class_name_words))

    # Lookup from child to parent.
    hierarchy = {}
    with open(input_dir / "hierarchy.txt") as fd:
        for line in fd:
            child_num, parent_num = line.split()
            child_num = clean_class_num(child_num)
            parent_num = clean_class_num(parent_num)

            assert child_num not in hierarchy
            hierarchy[child_num] = parent_num

    def get_hierarchical_class_name(class_num):
        tiers = [class_name_lookup[class_num]]

        while class_num in hierarchy:
            class_num = hierarchy[class_num]
            tiers.append(class_name_lookup[class_num])

        # All of them are 'Birds'
        tiers = [t for t in tiers if t != "Birds"]

        if DEV_ignore_perching_birds:
            tiers = [t for t in tiers if t != "Perching-Birds"]

        print(len(tiers))

        return "_".join(reversed(tiers))

    y = []
    for image_path in train_images:
        class_num, _ = image_path.split("/")
        if hierarchical:
            y.append(f"{class_num}_{get_hierarchical_class_name(class_num)}")
        else:
            y.append(f"{class_num}_{class_name_lookup[class_num]}")

    return x, y


def stratified_train_val_split(x, y, train_fraction: float):
    assert (
        0 < train_fraction < 1
    ), f"Must be a fraction between 0 and 1, not {train_fraction}!"

    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x, y, train_size=train_fraction, random_state=42, stratify=y
    )

    return x_train, y_train, x_val, y_val


def main():
    args = parse_args()

    output_dir = pathlib.Path(args.output)
    logger = helpers.create_logger("nabirds-split", output_dir)

    input_dir = pathlib.Path(args.input)
    x, y = load_train_data(
        input_dir,
        hierarchical=args.hierarchical,
        DEV_ignore_perching_birds=args.DEV_ignore_perching_birds,
    )

    dist = helpers.ClassDistribution(y)
    logger.info(
        "Class distribution: [min: %s, mean: %.2f, max: %s]",
        dist.min(),
        dist.mean(),
        dist.max(),
    )

    train_val_fraction = args.train_val
    x_train, y_train, x_val, y_val = stratified_train_val_split(
        x, y, train_val_fraction
    )
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
