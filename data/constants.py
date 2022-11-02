import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

data_mean_std = {
    "/mnt/10tb/data/inat21/resize-192": (
        torch.tensor([0.23754851520061493, 0.22912880778312683, 0.24746596813201904]),
        torch.tensor([0.4632684290409088, 0.48004600405693054, 0.37628623843193054]),
    ),
    "/local/scratch/cv_datasets/inat21/train_val_192": (
        torch.tensor([0.23754851520061493, 0.22912880778312683, 0.24746596813201904]),
        torch.tensor([0.4632684290409088, 0.48004600405693054, 0.37628623843193054]),
    ),
    "/mnt/10tb/data/inat21/resize-224": (
        torch.tensor([0.23762744665145874, 0.2292044311761856, 0.24757201969623566]),
        torch.tensor([0.4632636606693268, 0.48004215955734253, 0.37622377276420593]),
    ),
    "/mnt/10tb/data/inat21/resize-256": (
        torch.tensor([0.23768986761569977, 0.22925858199596405, 0.2476460039615631]),
        torch.tensor([0.4632672071456909, 0.480050653219223, 0.37618669867515564]),
    ),
    "/local/scratch/cv_datasets/inat21/resize-256": (
        torch.tensor([0.23768986761569977, 0.22925858199596405, 0.2476460039615631]),
        torch.tensor([0.4632672071456909, 0.480050653219223, 0.37618669867515564]),
    ),
    "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
}
