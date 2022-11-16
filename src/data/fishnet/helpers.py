import concurrent.futures
import os
import random

import cv2
import numpy as np
from tqdm.auto import tqdm


def extension_matches(name):
    return (
        name.endswith(".jpg")
        or name.endswith(".jpeg")
        or name.endswith(".png")
        or name.endswith(".gif")
    )


def stats_of(path, expected_channels):
    img = cv2.imread(path)
    width, height, actual_channels = img.shape
    assert actual_channels == expected_channels
    return (
        np.sum(img / 255, axis=(0, 1)),
        np.sum(img / 255 * img / 255, axis=(0, 1)),
        width * height,
    )


def load_statistics(directory, expected_channels=3, atol=1e-5):
    """
    Need to calculate mean and std for the individual channels so we can normalize the images.

    But we cannot put the whole thing in RAM, so we have to do it sequentially.

    Measure mean/std for 10, 31, 100, 316, 1K, 3162, 10K, etc until the we have two decimal points of accuracy. Then quit early.
    """

    total = np.zeros((expected_channels,), dtype=np.float64)
    total_squared = np.zeros((expected_channels,), dtype=np.float64)
    divisor = 0
    try:
        threadpool = concurrent.futures.ThreadPoolExecutor()
        filepaths = []
        for (dirpath, _, filenames) in os.walk(directory):
            for filename in filenames:
                if not extension_matches(filename):
                    continue

                filepaths.append(os.path.join(dirpath, filename))

        random.shuffle(filepaths)
        futures = [
            threadpool.submit(stats_of, path, expected_channels)
            for path in tqdm(filepaths)
        ]

        print(f"Submitted {len(futures)} calls to stats_of().")
        prev_mean = None
        prev_std = None
        for i, future in enumerate(
            tqdm(concurrent.futures.as_completed(futures), total=len(futures))
        ):
            sums, sums_squared, pixels = future.result()
            total += sums
            total_squared += sums_squared
            divisor += pixels

            if (i + 1) % 1000 == 0:
                mean = total / divisor
                var = (total_squared / divisor) - (mean * mean)
                std = np.sqrt(var)

                if (
                    prev_mean is not None
                    and prev_std is not None
                    and np.isclose(mean, prev_mean, atol=atol).all()
                    and np.isclose(std, prev_std, atol=atol).all()
                ):
                    print("Found it!")
                    print(mean, prev_mean, std, prev_std)
                    break
                else:
                    prev_mean = mean
                    prev_std = std
    finally:
        threadpool.shutdown(wait=False, cancel_futures=True)

    mean = total / divisor
    var = (total_squared / divisor) - (mean * mean)
    std = np.sqrt(var)

    return mean, std
