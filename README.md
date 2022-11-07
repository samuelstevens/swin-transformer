# Swin Transformer

[Link to original Swin Transformer project](https://github.com/microsoft/Swin-Transformer)

## Installation Instructions

1. Set up python packages

```sh
python -m venv venv
# Activate your virtual environment somehow
source venv/bin/activate.fish 
```

CUDA 11.6

```sh
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

CUDA 11.3

```sh
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Python packages

```sh
pip install matplotlib yacs timm einops black isort flake8 flake8-bugbear termcolor wandb preface opencv-python
```

2. Install Apex

Apex is not needed if you do not want to use fp16.

```sh
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


3. Download Data

We use the iNat21 dataseta available on [GitHub](https://github.com/visipedia/inat_comp/tree/master/2021)

```
cd /mnt/10tb
mkdir -p data/inat21
cd data/inat21
mkdir compressed raw
cd compressed
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz

# pv is just a progress bar
pv val.tar.gz | tar -xz
mv val ../raw/  # if I knew how tar worked I could have it extract to raw/

pv train.tar.gz | tar -xz
mv train ../raw/
```

4. Preprocess iNat 21

Use your root data folder and your size of choice.

```
export DATA_DIR=/mnt/10tb/data/inat21/
python -m data.inat preprocess $DATA_DIR val resize 192
python -m data.inat preprocess $DATA_DIR train resize 192
python -m data.inat preprocess $DATA_DIR val resize 256
python -m data.inat preprocess $DATA_DIR train resize 256
```

5. Login to Wandb

```
wandb login
```

6. Set up an `env.fish` file:

You need to provide `$VENV` and a `$RUN_OUTPUT` environment variables.
I recommend using a file to save these variables.

In fish:

```fish
# scripts/env.fish
set -gx VENV venv
set -gx RUN_OUTPUT /mnt/10tb/models/hierarchical-vision
```

Then run `source scripts/env.fish`

## AWS Helpers

Uninstall v1 of awscli:

```
sudo /usr/local/bin/pip uninstall awscli
```

Install v2:
```
cd ~/pkg
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install --bin-dir ~/.local/bin --install-dir ~/.local/aws-cli
```
