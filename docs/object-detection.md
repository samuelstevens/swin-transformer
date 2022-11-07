# Object Detection

## Get Started

1. Clone repository
2. Download data
3. Run experiments

### 1. Clone Repository

```sh
cd swin-transformer/

git clone https://github.com/samuelstevens/swin-transformer-object-detection.git object-detection

cd object-detection

pip install --verbose --editable .
```

### 2. Download Data

Datasets used:

* [fishnet](https://fishnet.ai)

```sh
# Make root directory
mkdir -p data/fishnet/v100
cd data/fishnet/v100

mkdir -p compressed raw

cd compressed
wget https://fishnet-data.s3-us-west-1.amazonaws.com/foid_labels_v100.zip
wget https://fishnet-data.s3-us-west-1.amazonaws.com/foid_images_v100.zip

unzip foid_labels_v100.zip
unzip foid_images_v100.zip

mv foid_labels_v100.csv images README.txt ../raw
rm -r __MACOSX

cd ..
tree -L 2 
```

The output should be something like:

```sh
.
├── compressed
│   ├── foid_images_v100.zip
│   └── foid_labels_v100.zip
└── raw
    ├── foid_labels_v100.csv
    ├── images
    └── README.txt

3 directories, 4 files
```

Then run a script to convert fishnet to the COCO format:

```sh
python -m data.fishnet cocofy \
  ./data/fishnet/v100/raw/images \
  ./data/fishnet/v100/raw/foid_labels_v100.csv \
  ./data/fishnet/v100/coco
```

Now `tree -L 2 v100` should produce:

```
v100
├── coco
│   ├── test
│   ├── train
│   └── val
├── compressed
│   ├── foid_images_v100.zip
│   └── foid_labels_v100.zip
└── raw
    ├── foid_labels_v100.csv
    ├── images
    └── README.txt

7 directories, 4 files
```

And in `v100/coco/<split>`, there will be many images and some `annotations.json` files.


### 3. Run Experiments

```
bash object-detection/tools/dist_train.sh \
  object-detection/configs/hierarchical-vision-project/blazing_blackberry_fishnet.py \
  <GPUS> \
  --seed 42 \
  --auto-resume \
  --diff-seed
```

This will write do the directory specified in the config file's `work_dir` location.
