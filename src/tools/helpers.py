import collections
import logging
import pathlib
import shutil
import statistics
import sys

from .. import concurrency


class ClassDistribution:
    def __init__(self, seq):
        self.counts = collections.Counter(seq)

    def min(self):
        return self.counts.most_common()[-1]

    def max(self):
        return self.counts.most_common(1)[0]

    def mean(self):
        return statistics.mean(self.counts.values())


def create_logger(name, output_dir: pathlib.Path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = "[%(asctime)s] [%(filename)s:%(lineno)d %(levelname)s] %(message)s"
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "log.txt", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger.info("Made directory %s", output_dir)

    return logger


def save_data(
    image_paths: list[pathlib.Path], classes: list[str], output_dir: pathlib.Path
) -> None:
    """
    Copies the images from image_paths into the output_dir.

    Expecrts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def output_path_of(i):
        image_path = image_paths[i]
        cls = classes[i]
        *_, filename = image_path.parts
        return output_dir / cls / filename

    try:
        pool = concurrency.BoundedExecutor()
        for i, path in enumerate(image_paths):
            pool.submit(output_path_of(i).parent.mkdir, parents=True, exist_ok=True)
        pool.finish(desc="Making directories")

        for i, path in enumerate(image_paths):
            pool.submit(shutil.copy2, str(path), output_path_of(i))
        pool.finish(desc="Copying data")
    finally:
        pool.shutdown()
