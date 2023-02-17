import logging
import pathlib
import sys


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
