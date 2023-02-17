"""
Makes a pretrain/downstream split by choosings X% of the species as "downstream".
"""
import argparse
import pathlib
import random
import shutil
import typing

from tqdm.auto import tqdm

from .. import concurrency
from . import helpers

T = typing.TypeVar("T")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory. Should contain train/ and val/ directories.",
    )
    parser.add_argument(
        "--frac",
        required=True,
        type=float,
        help="Proportion of input classes to use as a downstream task. Must be between 0 and 1.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will contain pretrain/ and downstream/ directories.",
    )
    # Optional arguments
    parser.add_argument("--seed", default=42, help="Random seed.")
    return parser.parse_args()


def get_classes(input_dir: pathlib.Path) -> set[str]:
    """
    Return a union of all classes in the train and validation set.
    """
    train_classes = {cls.stem for cls in sorted((input_dir / "train").iterdir())}
    val_classes = {cls.stem for cls in sorted((input_dir / "val").iterdir())}

    return train_classes | val_classes


def sample(classes: set[T], fraction: float, seed: int) -> set[T]:
    # round down to the nearest integer
    k = int(len(classes) * fraction)
    random.seed(seed)
    # Need to sort the classes first to convert from set (random order) to
    # list (fixed order) so that it reliably generates the same split.
    return set(random.sample(sorted(classes), k))


def copy_data(input_dir: pathlib.Path, output_dir: pathlib.Path, classes: set[str]):
    """
    input_dir has a train and a val directory.
    input_dir/<split> and input_dir/<split> both have class directories.
    input_dir/<split>/<CLS>/ has images.

    We add a train and val directory to output_dir, then mirror the structure of
    input_dir in output_dir. The only difference is that we only copy classes from
    input_dir that are in the classes argument.
    """

    # Helper function to get the output class
    # dir path for a given input class dir path.
    def output_dir_of(input_dir: pathlib.Path):
        *_, split, cls = input_dir.parts
        return output_dir / split / cls

    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)

    # We use a BoundedExecutor as a thin wrapper over
    # concurrent.futures.ThreadPoolExecutor, which lets us submit many "tasks"
    # to the operating system, which will run in parallel using different threads.
    # Using threads in this manner means we can copy data in parallel, which
    # massively speeds up copying, which is important for iNat21 (2.7M images)
    try:
        pool = concurrency.BoundedExecutor()

        def copy_data_split(split):
            """
            Copies all data for a given split (val, train).
            Made into a function so I don't duplicate it.
            """
            for class_dir in tqdm(sorted((input_dir / split).iterdir())):
                # Skip directories that aren't in classes
                if class_dir.name not in classes:
                    continue

                pool.submit(
                    shutil.copytree,
                    str(class_dir),
                    output_dir_of(class_dir),
                    dirs_exist_ok=False,
                )

        # Do validation first because it's faster.
        copy_data_split("val")
        copy_data_split("train")

        pool.finish(desc="Copying data")
    finally:
        pool.shutdown()


def main():
    args = parse_args()

    output_dir = pathlib.Path(args.output)
    logger = helpers.create_logger("inat21-split", output_dir)

    input_dir = pathlib.Path(args.input)
    all_classes = get_classes(input_dir)
    logger.info("Found %d input classes.", len(all_classes))

    fraction = args.frac
    seed = args.seed
    downstream_classes = sample(all_classes, fraction, seed)
    pretrain_classes = all_classes - downstream_classes
    logger.info(
        "Split into pretrain/downstream. Randomly sampled %d%% of the species in %s as our downstream task. The remaining %d%% are for pretraining.",
        int(fraction * 100),
        input_dir,
        int((1 - fraction) * 100),
    )
    logger.info("Last downstream class: %s", sorted(downstream_classes)[-1])
    logger.info("Last pretrain class: %s", sorted(pretrain_classes)[-1])

    # Do downstream first because it's faster.
    copy_data(input_dir, output_dir / "downstream", downstream_classes)
    copy_data(input_dir, output_dir / "pretrain", pretrain_classes)
    logger.info("Done.")


if __name__ == "__main__":
    main()
