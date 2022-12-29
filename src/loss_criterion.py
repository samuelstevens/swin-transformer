import timm.loss
import torch

from . import hierarchical



def build_loss_criterion(config):
    if config.AUG.MIXUP == 0 and config.MODEL.LABEL_SMOOTHING > 0.0:
        if config.HIERARCHICAL:
            raise NotImplementedError(
                "We don't support hierarchical loss with label smoothing and no mixup."
            )

        criterion = timm.loss.LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING
        )
    elif config.HIERARCHICAL:
        # If we have mixup, smoothing is handled with mixup label transform
        criterion = hierarchical.HierarchicalLoss(
            coeffs=config.TRAIN.HIERARCHICAL_COEFFS
        ).to(torch.cuda.current_device())
        # criterion = hierarchical.HierarchicalCrossEntropyLoss(
        #     coeffs=config.TRAIN.HIERARCHICAL_COEFFS
        # ).to(torch.cuda.current_device())
    else:
        criterion = torch.nn.NLLLoss()
        # criterion = torch.nn.CrossEntropyLoss()

    return criterion
