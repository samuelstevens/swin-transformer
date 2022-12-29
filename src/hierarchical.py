import einops
import torch
import torch.nn as nn


def accuracy(output, target, topk=(1,), hierarchy_level=-1):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Copied from rwightman/pytorch-image-models/timm/utils/metrics.py and modified
    to work with hierarchical outputs as well.

    When the output is hierarchical, only returns the accuracy for `hierarchy_level`
    (default -1, which is the fine-grained level).
    """
    output_levels = 1
    if isinstance(output, list):
        output_levels = len(output)
        output = output[-1]

    batch_size = output.size(0)

    # Target might have multiple levels because of the hierarchy
    if target.squeeze().ndim == 2:
        assert target.squeeze().shape == (batch_size, output_levels)
        target = target[:, -1]

    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size
        for k in topk
    ]


class LSR(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()

        self.reduction = reduction

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        loss = torch.sum(- x * target, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class FineGrainedLoss(torch.nn.NLLLoss):
    """
    A cross-entropy used with hierarchical inputs and targets and only
    looks at the finest-grained tier (the last level).
    """

    def forward(self, inputs, targets):
        fine_grained_inputs = inputs[-1]
        fine_grained_targets = targets[:, -1]
        return super().forward(fine_grained_inputs, fine_grained_targets)

class HierarchicalLoss(LSR):
    def __init__(self, *args, coeffs=(1.0,), **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.clone().detach().type(torch.float)
        else:
            coeffs = torch.tensor(coeffs, dtype=torch.float)

        self.register_buffer("coeffs", coeffs)

    def forward(self, inputs, targets):

        if not isinstance(targets, list):
            targets = einops.rearrange(targets, "batch tiers -> tiers batch")

        assert (
            len(inputs) == len(targets) == len(self.coeffs)
        ), f"{len(inputs)} != {len(targets)} != {len(self.coeffs)}"

        losses = torch.stack(
            [
                # Need to specify arguments to super() because of some a bug
                # with super() in list comprehensions/generators (unclear)
                super(HierarchicalLoss, self).forward(input, target)
                for input, target in zip(inputs, targets)

            ]
        )

        return torch.dot(self.coeffs, losses)