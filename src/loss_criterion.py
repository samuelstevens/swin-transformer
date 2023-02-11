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
        criterion = hierarchical.HierarchicalCrossEntropyLoss(
            coeffs=config.TRAIN.HIERARCHICAL_COEFFS
        ).to(torch.cuda.current_device())
    else:
        criterion = torch.nn.CrossEntropyLoss()

    return criterion

# script for loss function hot-ice
def build_loss_criterion_hot_ice(config, hierarchy, classes, weights):
    if config.AUG.MIXUP == 0 and config.MODEL.LABEL_SMOOTHING > 0.0:
        if config.HIERARCHICAL:
            raise NotImplementedError(
                "We don't support hierarchical loss with label smoothing and no mixup."
            )

        criterion = timm.loss.LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING
        )
    elif config.HIERARCHICAL:
        criterion = hierarchical.HierarchicalCrossEntropyLoss_hot_ice(hierarchy, classes, weights).to(torch.cuda.current_device())
    else:
        criterion = torch.nn.CrossEntropyLoss()

    return criterion

# script for loss function red-chilli
def build_loss_criterion_red_chilli(config, hierarchy, classes, weights):
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

        criterion = hierarchical.HierarchicalCrossEntropyLoss_red_chilli(hierarchy, classes, weights,
            coeffs=config.TRAIN.HIERARCHICAL_COEFFS
        ).to(torch.cuda.current_device())
    else:
        criterion = torch.nn.CrossEntropyLoss()

    return criterion



