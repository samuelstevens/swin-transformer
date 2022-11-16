import concurrent.futures
import dataclasses
import json
import operator
import os
import pathlib
import shutil

from tqdm.auto import tqdm


@dataclasses.dataclass
class Image:
    # Relative to the train_val2019 folder.
    rel_path: pathlib.Path
    cls: str
    filename: str
    id: int

    def __init__(self, id: int, *, filepath: str, cls: str):
        self.id = id
        self.cls = cls

        # All images are redundantly prefixed with this string.
        prefix = "train_val2019/"
        assert filepath.startswith(prefix)
        self.rel_path = pathlib.Path(filepath.removeprefix(prefix))
        self.filename = self.rel_path.name

    def copy_to(self, parent, output_dir):
        shutil.copy2(parent / self.rel_path, output_dir / self.cls)


def organize(
    train_val_images_path,
    train_annotations_path,
    val_annotations_path,
    categories_path,
    output_path,
):
    """
    Make two trees:

    <output_path>
    |-- train
    |   |-- <class1>
    |   |   |-- <img1>
    |   |   |-- ...
    |   |   `-- <imgN>
    |   |-- ...
    |   `-- <classM>
    |       |-- <img1>
    |       |-- ...
    |       `-- <imgN>
    `-- val
        |-- <class1>
        |   |-- <img1>
        |   |-- ...
        |   `-- <imgN>
        |-- ...
        `-- <classM>
            |-- <img1>
            |-- ...
            `-- <imgN>
    """

    output_path = pathlib.Path(output_path)
    (output_path / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "train").mkdir(parents=True, exist_ok=True)

    train_val_images_path = pathlib.Path(train_val_images_path)

    with open(val_annotations_path) as fd:
        val_annotations = json.load(fd)

    with open(train_annotations_path) as fd:
        train_annotations = json.load(fd)

    with open(categories_path) as fd:
        category_annotations = json.load(fd)

    # Lookup from id to the evolutionary taxonomy.
    category_lookup: dict[int, str] = {}

    # Any weird categories
    oddities = []

    for category in category_annotations:
        genus, *species = category["name"].split()
        if len(species) > 1:
            oddities.append(category)

        assert genus == category["genus"]
        category_lookup[category["id"]] = "_".join(
            (
                category["kingdom"].title(),
                category["phylum"].title(),
                category["class"].title(),
                category["order"].title(),
                category["family"].title(),
                genus.title(),
                species[0].lower(),
            )
        )

    print(oddities)

    image_category: dict[int, int] = {}

    for annotation in val_annotations["annotations"]:
        image_category[annotation["image_id"]] = category_lookup[
            annotation["category_id"]
        ]

    for annotation in train_annotations["annotations"]:
        image_category[annotation["image_id"]] = category_lookup[
            annotation["category_id"]
        ]

    val_images = [
        Image(image["id"], filepath=image["file_name"], cls=image_category[image["id"]])
        for image in val_annotations["images"]
    ]
    train_images = [
        Image(image["id"], filepath=image["file_name"], cls=image_category[image["id"]])
        for image in train_annotations["images"]
    ]

    classes = {image.cls for image in val_images + train_images}
    for cls in tqdm(classes, desc="Making classes"):
        (output_path / "train" / cls).mkdir(parents=True, exist_ok=True)
        (output_path / "val" / cls).mkdir(parents=True, exist_ok=True)

    try:
        futures = []
        threadpool = concurrent.futures.ThreadPoolExecutor()

        for image in tqdm(val_images, desc="Validation images"):
            futures.append(
                threadpool.submit(
                    image.copy_to,
                    train_val_images_path,
                    output_path / "val",
                )
            )
        for image in tqdm(train_images, desc="Training images"):
            futures.append(
                threadpool.submit(
                    image.copy_to,
                    train_val_images_path,
                    output_path / "train",
                )
            )

        # Make sure all images have been properly copied.
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Copying images",
        ):
            future.result()

    finally:
        threadpool.shutdown(wait=False, cancel_futures=True)


class BoundedExecutor:
    def __init__(self, pool_cls=concurrent.futures.ThreadPoolExecutor):
        self._pool = pool_cls()
        self._futures = []

    def submit(self, *args, **kwargs):
        self._futures.append(self._pool.submit(*args, **kwargs))

    def shutdown(self, **kwargs):
        self._pool.shutdown(wait=False, cancel_futures=True, **kwargs)

    def finish(self, *, desc: str = ""):
        return [
            future.result()
            for future in tqdm(
                concurrent.futures.as_completed(self._futures),
                total=len(self._futures),
                desc=desc,
            )
        ]


def inat21(inat19_path, inat21_path, output_path):
    output_path = pathlib.Path(output_path)
    inat19_path = pathlib.Path(inat19_path)

    inat21_labels: dict[Label, int] = {}
    with os.scandir(inat21_path) as it:
        for entry in it:
            if entry.is_file():
                continue

            number, label = parse_label_with_number(entry.name)
            inat21_labels[label] = number

    inat19_labels: set[Label] = set()
    with os.scandir(inat19_path) as it:
        for entry in it:
            if entry.is_file():
                continue

            inat19_labels.add(parse_label(entry.name))

    try:
        pool = BoundedExecutor()
        label_map = []
        for label, number in sorted(inat21_labels.items(), key=operator.itemgetter(1)):
            if label not in inat19_labels:
                continue

            pool.submit(
                shutil.copytree,
                str(inat19_path / label.foldername()),
                str(output_path / "val" / label.foldername(number)),
                dirs_exist_ok=True,
            )
            label_map.append(number)

        with open("src/data/inat21_inat19_map.txt", "w") as fd:
            for inat21_label in label_map:
                fd.write(f"{inat21_label}\n")

        pool.finish(desc="Copying folders")
    finally:
        pool.shutdown()


@dataclasses.dataclass(frozen=True)
class Label:
    kingdom: str
    phylum: str
    cls: str
    order: str
    family: str
    genus: str
    species: str

    def foldername(self, number=None) -> str:
        parts = [
            self.kingdom.title(),
            self.phylum.title(),
            self.cls.title(),
            self.order.title(),
            self.family.title(),
            self.genus.title(),
            self.species.lower(),
        ]
        if number is not None:
            parts.insert(0, str(number).zfill(5))
        return "_".join(parts)

    def __eq__(self, other):
        if not isinstance(other, Label):
            return False

        return (
            self.kingdom.lower() == other.kingdom.lower()
            and self.phylum.lower() == other.phylum.lower()
            and self.cls.lower() == other.cls.lower()
            and self.order.lower() == other.order.lower()
            and self.family.lower() == other.family.lower()
            and self.genus.lower() == other.genus.lower()
            and self.species.lower() == other.species.lower()
        )


def parse_label_with_number(name: str) -> tuple[int, Label]:
    number, *labels = name.split("_")
    return int(number), Label(*labels)


def parse_label(name: str) -> Label:
    labels = name.split("_")
    return Label(*labels)
