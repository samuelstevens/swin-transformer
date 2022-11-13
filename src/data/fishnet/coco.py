import abc
import collections
import concurrent.futures
import csv
import json
import logging
import os
import pathlib
import random
import shutil
from typing import Any, Generic, Literal, TypeVar

import cv2  # type: ignore
import preface
from tqdm.auto import tqdm  # type: ignore

T = TypeVar("T")
SplitName = Literal["train", "val", "test"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def finish_all(futures: list[concurrent.futures.Future[T]]) -> list[T]:
    return [
        future.result()
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))
    ]


class JsonObj(abc.ABC):
    @abc.abstractmethod
    def asdict(self) -> dict[str, object]:
        raise NotImplementedError()


class IdLookup(Generic[T]):
    def __init__(self) -> None:
        self._lookup: dict[T, int] = {}

    def id_of(self, obj: T, insert: bool = True) -> int:
        if obj not in self._lookup and insert:
            self._lookup[obj] = len(self._lookup)

        return self._lookup[obj]

    def contains(self, obj: T) -> bool:
        return obj in self._lookup


class BoundingBox:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> "BoundingBox":
        x = max(0, float(dct["x_min"]))
        y = max(0, float(dct["y_min"]))

        width = max(0, float(dct["x_max"]) - x)
        height = max(0, float(dct["y_max"]) - y)

        return cls(x, y, width, height)


class Annotation(JsonObj):
    def __init__(
        self, id: int, image_id: int, category_id: int, bbox: BoundingBox
    ) -> None:
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox

    def asdict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": [self.bbox.x, self.bbox.y, self.bbox.width, self.bbox.height],
            "area": self.bbox.area,
            "iscrowd": 0,
        }


class Image(JsonObj):
    def __init__(self, path: str | pathlib.Path, id: int, split: SplitName):
        self.id = id
        self.path = str(path)
        self.split = split

        self.file_name = os.path.basename(path)
        self.height = None
        self.width = None
        self.annotations = []

    def init(self) -> "Image":
        """Call this method when you want to actually open
        the path and load the height and width (expensive).
        """
        height, width, channels = cv2.imread(self.path).shape
        assert channels == 3

        self.height = height
        self.width = width

    def add_annotation(self, ann: Annotation):
        assert self.id == ann.image_id, f"{self.id} != {ann.image_id}"
        self.annotations.append(ann)

    def asdict(self) -> dict[str, object]:
        if self.height is None or self.width is None:
            raise RuntimeError("You must call .init() before .asdict().")

        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "file_name": self.file_name,
        }


class Category(JsonObj):
    def __init__(self, id: int, name: str, supercategory: str) -> None:
        self.id = id
        self.name = name
        self.supercategory = supercategory

    def asdict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }


class Split(JsonObj):
    def __init__(self) -> None:
        self.images: list[Image] = []
        self._image_ids: set[int] = set()

        self.categories: list[Category] = []
        self._category_ids: set[int] = set()

    def add_image(self, image: Image) -> None:
        if image.id in self._image_ids:
            return

        self._image_ids.add(image.id)
        self.images.append(image)

    def add_category(self, category: Category) -> None:
        if category.id in self._category_ids:
            return

        self._category_ids.add(category.id)
        self.categories.append(category)

    def asdict(self) -> dict[str, object]:
        annotations = preface.flattened(image.annotations for image in self.images)

        for ann in annotations:
            if ann.category_id not in self._category_ids:
                raise RuntimeError(
                    f"Annotation {ann.id}'s category id {ann.category_id} is unseen."
                )

            if ann.image_id not in self._image_ids:
                raise RuntimeError(
                    f"Annotation {ann.id}'s image id {ann.image_id} is unseen."
                )

        return {
            "images": [image.asdict() for image in self.images],
            "annotations": [ann.asdict() for ann in annotations],
            "categories": [category.asdict() for category in self.categories],
        }


class Dataset:
    def __init__(self) -> None:
        self.train = Split()
        self.val = Split()
        self.test = Split()

    def __getitem__(self, key: SplitName) -> Split:
        if key == "train":
            return self.train
        elif key == "val":
            return self.val
        elif key == "test":
            return self.test
        else:
            preface.never()

    def splits(self) -> list[tuple[SplitName, Split]]:
        return [("train", self.train), ("val", self.val), ("test", self.test)]


def parse_split(row: dict[str, object]) -> Literal["train", "val", "test"]:
    def eval_bool(obj: object) -> bool:
        if obj == "True":
            return True
        elif obj == "False":
            return False
        else:
            raise RuntimeError(obj)

    train = eval_bool(row["train"])
    val = eval_bool(row["val"])
    test = eval_bool(row["test"])

    if train and not val and not test:
        return "train"
    elif not train and val and not test:
        return "val"
    elif not train and not val and test:
        return "test"
    else:
        raise RuntimeError(train, val, test)


def calc_limit(limit: int, size: float) -> int:
    # Do this int(size) == size check because size will always be a float
    if int(size) == size:
        return size

    if size > 1.0:
        logger.warn(
            "size is floating point and > 1, which means more than 100% of the data. [size: %s]",
            size,
        )

    return int(size * limit)


def cocofy(
    image_dir: str | pathlib.Path,
    label_file: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    size: float | None,
) -> None:
    """
    Arguments should all be absolute paths
    """
    image_dir = pathlib.Path(image_dir)
    output_dir = pathlib.Path(output_dir)

    # Lookup from filename to id
    image_id_lookup = IdLookup[str]()

    # Lookup from split to image id to Image object
    images = collections.defaultdict(dict)

    # Lookup from category name to id
    category_id_lookup = IdLookup[str]()

    dataset = Dataset()

    for split, _ in dataset.splits():
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True, parents=True)

    with open(label_file) as fd:
        # Load all annotations, images and categories
        for i, row in enumerate(tqdm(csv.DictReader(fd), desc="Reading labels")):
            split = parse_split(row)

            # Set up image
            image_uuid = row["img_id"]
            image_id = image_id_lookup.id_of(image_uuid)
            if image_id not in images[split]:
                image_path = image_dir / f"{image_uuid}.jpg"
                images[split][image_id] = Image(image_path, image_id, split)
            image = images[split][image_id]

            # Set up category
            category_name = row["label_l1"]
            category = Category(
                category_id_lookup.id_of(category_name), category_name, row["label_l2"]
            )
            # All splits need all categories
            for split, _ in dataset.splits():
                dataset[split].add_category(category)

            # Set up annotation
            annotation = Annotation(
                i, image.id, category.id, BoundingBox.from_dict(row)
            )
            image.add_annotation(annotation)

    # Calculate limits
    train_limit = calc_limit(len(images["train"]), size)
    train_image_ids = list(images["train"])
    random.seed(42)
    random.shuffle(train_image_ids)
    train_image_ids = train_image_ids[:train_limit]
    for image_id in set(images["train"]) - set(train_image_ids):
        del images["train"][image_id]

    # Add training images to splits
    for split, annotations in dataset.splits():
        for image in images[split].values():
            annotations.add_image(image)

    try:
        threadpool = concurrent.futures.ThreadPoolExecutor()

        copy_futures = []
        init_futures = []

        for split, annotations in dataset.splits():
            for image in annotations.images:
                copy_futures.append(
                    threadpool.submit(
                        shutil.copy2, image.path, output_dir / split / image.file_name
                    )
                )
                init_futures.append(threadpool.submit(image.init))

        finish_all(init_futures)

        # Write annotation files
        for split, annotations in dataset.splits():
            with open(output_dir / split / "annotations.json", "w") as fd:
                json.dump(annotations.asdict(), fd)

        finish_all(copy_futures)
    finally:
        threadpool.shutdown(wait=False, cancel_futures=True)
