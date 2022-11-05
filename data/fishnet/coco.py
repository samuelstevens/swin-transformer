import abc
import concurrent.futures
import csv
import json
import logging
import os
import pathlib
import shutil
from typing import Any, Generic, Literal, TypedDict, TypeVar

import cv2  # type: ignore
import preface
from tqdm.auto import tqdm  # type: ignore

T = TypeVar("T")
SplitName = Literal["train", "val", "test"]

logger = logging.getLogger(__name__)


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


class Image(JsonObj):
    def __init__(self, path: str, id: int):
        self.id = id
        self.path = path
        self.file_name = os.path.basename(path)

        height, width, channels = cv2.imread(path).shape
        assert channels == 3

        self.height = height
        self.width = width

    def asdict(self) -> dict[str, object]:
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
            "segmentation": [],
            "bbox": [self.bbox.x, self.bbox.y, self.bbox.width, self.bbox.height],
            "area": self.bbox.area,
            "iscrowd": 0,
        }


class Split(JsonObj):
    def __init__(self) -> None:
        self.images: list[Image] = []
        self._image_ids: set[int] = set()

        self.categories: list[Category] = []
        self._category_ids: set[int] = set()

        self.annotations: list[Annotation] = []
        self._annotation_ids: set[int] = set()

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

    def add_annotation(self, annotation: Annotation) -> None:
        if annotation.id in self._annotation_ids:
            logger.warn("Tried to add duplicate annotation. [id: %s]", annotation.id)
            return

        assert (
            annotation.image_id in self._image_ids
        ), f"Image {annotation.image_id} does not exist!"
        assert (
            annotation.category_id in self._category_ids
        ), f"Category {annotation.category_id} does not exist!"

        self._annotation_ids.add(annotation.id)
        self.annotations.append(annotation)

    def asdict(self) -> dict[str, object]:
        return {
            "images": [image.asdict() for image in self.images],
            "annotations": [annotation.asdict() for annotation in self.annotations],
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


def cocofy(image_dir: str, label_file: str, output_dir: str | pathlib.Path) -> None:
    """
    Arguments should all be absolute paths
    """
    output_dir = pathlib.Path(output_dir)

    # Lookup from filename to id
    image_id_lookup = IdLookup[str]()

    # Lookup from category name to id
    category_id_lookup = IdLookup[str]()

    dataset = Dataset()

    for split, annotations in dataset.splits():
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True, parents=True)

    try:
        threadpool = concurrent.futures.ThreadPoolExecutor()
        fd = open(label_file)

        image_futures = {}

        for entry in tqdm(os.scandir(image_dir), desc="Scanning directory"):
            uuid, ext = os.path.splitext(entry.name)
            if ext != ".jpg":
                logger.warn(
                    f"File  has a bad extension. Skipping. [file: '%s', ext: '%s']",
                    entry.name,
                    ext,
                )
                continue

            image_id = image_id_lookup.id_of(uuid)

            image_futures[image_id] = threadpool.submit(Image, entry.path, image_id)

        csvreader = csv.DictReader(fd)

        copy_futures = []
        logger.info(
            "This will block with no progress for ~1 minute as we wait for images."
        )
        for i, row in enumerate(tqdm(csvreader, desc="Reading labels")):
            image_id = image_id_lookup.id_of(row["img_id"], insert=False)

            category_name = row["label_l1"]
            category = Category(
                category_id_lookup.id_of(category_name), category_name, row["label_l2"]
            )

            annotation = Annotation(
                i, image_id, category.id, BoundingBox.from_dict(row)
            )

            split = parse_split(row)
            # This line can block for quite a while. But while it is blocking,
            # other images are loading.
            image = image_futures[image_id].result(timeout=None)

            dataset[split].add_category(category)
            dataset[split].add_image(image)
            dataset[split].add_annotation(annotation)

            # Copy images using existing threadpool (non-blocking call)
            copy_futures.append(
                threadpool.submit(
                    shutil.copy2, image.path, output_dir / split / image.file_name
                )
            )

        # Write annotation files
        for split, annotations in dataset.splits():
            with open(split_dir / "annotations.json", "w") as fd:
                json.dump(annotations.asdict(), fd)

        # Make sure all images have been properly copied.
        for future in tqdm(
            concurrent.futures.as_completed(copy_futures),
            total=len(copy_futures),
            desc="Copying images",
        ):
            future.result()
    finally:
        fd.close()
        threadpool.shutdown(wait=False, cancel_futures=True)
