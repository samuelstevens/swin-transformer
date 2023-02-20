import torch.nn as nn

import src.config as config
from src.models import build_model
from src.tools.class_patterns import build_loader

from src.models.swin_transformer_v2 import SwinTransformerV2, HierarchicalHead

from src.externals.pytorch_grad_cam import GradCAM
from src.externals.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.externals.pytorch_grad_cam.utils.image import show_cam_on_image

class SwinWrapperForCam(nn.Module):
    def __init__(self, swin_transformer):
        super().__init__()
        assert isinstance(swin_transformer, SwinTransformerV2) #! Assumes underlying SwinTransformerV2 model
        self.swin = swin_transformer

    def forward(self, x):
        features = self.swin.forward_features(x)
        if isinstance(self.swin.head, HierarchicalHead):
            #! Only use last head if hierarchical-based model
            out = self.swin.head.heads[6](features)
        else:
            out = self.swin.head(features)
        return out


    def get_target_layers(self):
        """
            If you want to try other layers, I would recommend inheriting this
            class and overloading this method.
        """
        return [self.swin.layers[-1].blocks[-1].norm1]

def reshape_transform(tensor, height=6, width=6):
    """
    The default parameters are set up for the specific image size currently used.
    This assumes 192 x 192 img size. This will have to be changed if the number
    of patches and/or image size is changed.
    """
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def run_grad_cam(model, img_tensor, cls_tgt):
    cam = GradCAM(model=model, target_layers=model.get_target_layers(), reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=img_tensor, targets=cls_tgt)
    return grayscale_cam[0, :]


def saliency(model, img_tensor, cls_id, cam_type="gradcam"):
    cls_tgt = [ClassifierOutputTarget(cls_id)]
    model_for_cam = SwinWrapperForCam(model)
    if cam_type == "gradcam":
        return run_grad_cam(model_for_cam, img_tensor, cls_tgt)
    
    raise NotImplementedError(f"cam_type '{cam_type}' has not been implemented yet")

if __name__ == "__main__":
    from argparse import ArgumentParser

    import torch
    import torch.nn as nn

    import numpy as np

    from PIL import Image

    import src.config as config
    from src.models import build_model
    from src.tools.class_patterns import build_loader, image_from_dataset

    # Options
    #! Note red-chilli really is single head, but has weights for all heads?
    options = {
        "groovy-grape" : ("/local/scratch/hierarchical-vision-checkpoints/groovy-grape-192-epoch89.pth", "multi-head"),
        "fuzzy-fig" : ("/local/scratch/hierarchical-vision-checkpoints/fuzzy-fig-192-epoch89.pth", "single-head"),
        "red-chilli" : ("/local/scratch/hierarchical-vision-checkpoints/red-chilli-192-epoch89.pth", "multi-head")
    }

    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="groovy-grape", choices=options.keys())
    args = parser.parse_args()

    cfg = config._C.clone()
    cfg.defrost()
    cfg.DATA.DATA_PATH = "/local/scratch/cv_datasets/inat21/resize-192"
    cfg.DATA.DATASET = "inat21"
    cfg.DATA.IMG_SIZE = 192
    cfg.MODEL.TYPE = "swinv2"
    cfg.MODEL.NAME = "swinv2_base_window12"
    cfg.MODEL.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINV2.EMBED_DIM = 128
    cfg.MODEL.SWINV2.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.SWINV2.NUM_HEADS = [4, 8, 16, 32]
    cfg.MODEL.SWINV2.WINDOW_SIZE = 12
    cfg.MODEL.HIERARCHICAL_COEFFS = []
    cfg.TRAIN.DEVICE_BATCH_SIZE = 1
    cfg.TEST.SEQUENTIAL = True
    cfg.HIERARCHICAL = True if options[args.mode][1] == "multi-head" else False
    cfg.MODE = "eval"
    cfg.freeze()

    dataset, dataloader = build_loader(cfg)
    model = build_model(cfg)

    #! Load weights
    checkpoint = torch.load(options[args.mode][0], map_location="cpu")
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=True)
    
    IMG_ID = 6995

    img_ten, cls_id = dataset[IMG_ID]
    if options[args.mode][1] == "multi-head":
        cls_id = cls_id[-1]

    # Notice that the img_ten is a preprocessed tensor of batch_size one
    grayscale_cam = saliency(model, img_ten.unsqueeze(0), cls_id, cam_type="gradcam")

    # Notice that we have to give it the rgb img b/w 0 & 1 of the original input tensor
    rgb_img = np.array(image_from_dataset(dataset, IMG_ID)).astype(np.float32) / 255
    vis_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    Image.fromarray(vis_img).save("saliency.png")