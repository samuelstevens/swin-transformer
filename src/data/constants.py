import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

data_mean_std = {
    "/mnt/10tb/data/inat21/resize-192": (
        torch.tensor([0.4632684290409088, 0.48004600405693054, 0.37628623843193054]),
        torch.tensor([0.23754851520061493, 0.22912880778312683, 0.24746596813201904]),
    ),
    "/mnt/10tb/data/inat19/inat21": (
        torch.tensor([0.4632684290409088, 0.48004600405693054, 0.37628623843193054]),
        torch.tensor([0.23754851520061493, 0.22912880778312683, 0.24746596813201904]),
    ),
    "/local/scratch/cv_datasets/inat21/resize-192": (
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
    "/local/scratch/stevens.994/data/tiger-beetle/processed": (
        torch.tensor([0.503083218844332, 0.5775137087338068, 0.6259513527945232]),
        torch.tensor([0.2518764425218349, 0.23707872616648928, 0.2373721641220995]),
    ),
    "/local/scratch/paul.1164/data/inat21/raw": (
        torch.tensor([0.4632684290409088, 0.48004600405693054, 0.37628623843193054]),
        torch.tensor([0.23754851520061493, 0.22912880778312683, 0.24746596813201904]),
    ),
    "/local/scratch/paul.1164/data/nabird/raw": (
        torch.tensor([0.49044106, 0.5076765, 0.46390218]),
        torch.tensor([0.16689847, 0.1688618, 0.18529404]),
    ),
    "/local/scratch/paul.1164/data/ip102/raw": (
        torch.tensor([0.51354748, 0.54016679, 0.38778601]),
        torch.tensor([0.19195388, 0.19070604, 0.19121135]),
    ),
    "/local/scratch/paul.1164/data/stanford_dogs/raw": (
        torch.tensor([0.47468446, 0.43789249, 0.38218195]),
        torch.tensor([0.22826478, 0.22320983, 0.21866826]),
    ),
    "/local/scratch/cv_datasets/ip102/classification/preprocessed": (
        torch.tensor([0.5139346122741699, 0.5405563116073608, 0.3881925940513611]),
        torch.tensor([0.2663924992084503, 0.2535426616668701, 0.285331666469574]),
    ),
    "/local/scratch/cv_datasets/nabirds/full-data-hierarchical": (
        torch.tensor([0.49161002039909363, 0.508598268032074, 0.4649352729320526]),
        torch.tensor([0.21895255148410797, 0.21770493686199188, 0.264631450176239]),
    ),
    "/local/scratch/cv_datasets/nabirds/0.1-data": (
        torch.tensor([0.491899311542511, 0.5088262557983398, 0.4643000364303589]),
        torch.tensor([0.21911440789699554, 0.21796336770057678, 0.26489198207855225]),
    ),
    "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
}
