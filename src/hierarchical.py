import einops
import torch

import numpy as np
from typing import List
from nltk.tree import Tree

from .utils import (
    get_label,
    )


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


class FineGrainedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    A cross-entropy used with hierarchical inputs and targets and only
    looks at the finest-grained tier (the last level).
    """

    def forward(self, inputs, targets):
        fine_grained_inputs = inputs[-1]
        fine_grained_targets = targets[:, -1]
        return super().forward(fine_grained_inputs, fine_grained_targets)


class HierarchicalCrossEntropyLoss(torch.nn.CrossEntropyLoss):
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
                super(HierarchicalCrossEntropyLoss, self).forward(input, target)
                for input, target in zip(inputs, targets)
            ]
        )

        return torch.dot(self.coeffs, losses)



# script for hot-ice
class HierarchicalLLLoss_hot_ice(torch.nn.Module):
    """
    Hierachical log likelihood loss.

    The weights must be implemented as a nltk.tree object and each node must
    be a float which corresponds to the weight associated with the edge going
    from that node to its parent. The value at the origin is not used and the
    shapre of the weight tree must be the same as the associated hierarchy.

    The input is a flat probability vector in which each entry corresponds to
    a leaf node of the tree. We use alphabetical ordering on the leaf nodes
    labels, which corresponds to the 'normal' imagenet ordering.

    Args:
        hierarchy: The hierarchy used to define the loss.
        classes: A list of classes defining the order of the leaf nodes.
        weights: The weights as a tree of similar shape as hierarchy.
    """

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HierarchicalLLLoss_hot_ice, self).__init__()

        assert hierarchy.treepositions() == weights.treepositions()

        # the tree positions of all the leaves
        positions_leaves = {get_label(hierarchy[p]): p for p in hierarchy.treepositions("leaves")}
        num_classes = len(positions_leaves)

        # we use classes in the given order 
        positions_leaves = [positions_leaves[c] for c in classes]

        # the tree positions of all the edges (we use the bottom node position)
        positions_edges = hierarchy.treepositions()[1:]  # the first one is the origin

        # map from position tuples to leaf/edge indices
        index_map_leaves = {positions_leaves[i]: i for i in range(len(positions_leaves))}
        index_map_edges = {positions_edges[i]: i for i in range(len(positions_edges))}

        # edge indices corresponding to the path from each index to the root
        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), 0, -1)] for position in positions_leaves]

        # get max size for the number of edges to the root
        num_edges = max([len(p) for p in edges_from_leaf])

        # helper that returns all leaf positions from another position wrt to the original position
        def get_leaf_positions(position):
            node = hierarchy[position]
            if isinstance(node, Tree):
                return node.treepositions("leaves")
            else:
                return [()]

        # indices of all leaf nodes for each edge index
        leaf_indices = [[index_map_leaves[position + leaf] for leaf in get_leaf_positions(position)] for position in positions_edges]

        # save all relevant information as pytorch tensors for computing the loss on the gpu
        self.onehot_den = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.onehot_num = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.weights = torch.nn.Parameter(torch.zeros([num_classes, num_edges]), requires_grad=False)

        # one hot encoding of the numerators and denominators and store weights
        for i in range(num_classes):
            for j, k in enumerate(edges_from_leaf[i]):
                self.onehot_num[i, leaf_indices[k], j] = 1.0
                self.weights[i, j] = get_label(weights[positions_edges[k]])
            for j, k in enumerate(edges_from_leaf[i][1:]):
                self.onehot_den[i, leaf_indices[k], j] = 1.0
            self.onehot_den[i, :, j + 1] = 1.0  # the last denominator is the sum of all leaves

    def forward(self, inputs, target):
        """
        Foward pass, computing the loss.

        Args:
            inputs: Class _probabilities_ ordered as the input hierarchy.
            target: The index of the ground truth class.
        """


        # add a sweet dimension to inputs
        inputs = torch.unsqueeze(inputs, 1)

        # sum of probabilities for numerators
        num = torch.squeeze(torch.bmm(inputs, self.onehot_num[target]))
        # sum of probabilities for denominators
        den = torch.squeeze(torch.bmm(inputs, self.onehot_den[target]))
        # compute the neg logs for non zero numerators and store in there
        idx = num != 0
        num[idx] = -torch.log(num[idx] / den[idx])
        # weighted sum of all logs for each path (we flip because it is numerically more stable)
        num = torch.sum(torch.flip(self.weights[target] * num, dims=[1]), dim=1)
        # return sum of losses / batch size
        return torch.mean(num)

class HierarchicalCrossEntropyLoss_hot_ice(HierarchicalLLLoss_hot_ice):
    """
    Combines softmax with HierachicalNLLLoss. Note that the softmax is flat.
    """

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HierarchicalCrossEntropyLoss_hot_ice, self).__init__(hierarchy, classes, weights)

    def forward(self, inputs, index):
        return super(HierarchicalCrossEntropyLoss_hot_ice, self).forward(torch.nn.functional.softmax(inputs, 1), index)


# Script for red-chilli hierarchical loss
class HierarchicalLLLoss_red_chilli(torch.nn.Module):
    """
    Hierachical log likelihood loss.

    The weights must be implemented as a nltk.tree object and each node must
    be a float which corresponds to the weight associated with the edge going
    from that node to its parent. The value at the origin is not used and the
    shapre of the weight tree must be the same as the associated hierarchy.

    The input is a flat probability vector in which each entry corresponds to
    a leaf node of the tree. We use alphabetical ordering on the leaf nodes
    labels, which corresponds to the 'normal' imagenet ordering.

    Args:
        hierarchy: The hierarchy used to define the loss.
        classes: A list of classes defining the order of the leaf nodes.
        weights: The weights as a tree of similar shape as hierarchy.
    """

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HierarchicalLLLoss_red_chilli, self).__init__()

        assert hierarchy.treepositions() == weights.treepositions()

        # the tree positions of all the leaves
        positions_leaves = {get_label(hierarchy[p]): p for p in hierarchy.treepositions("leaves")}
        num_classes = len(positions_leaves)

        # we use classes in the given order 
        positions_leaves = [positions_leaves[c] for c in classes]

        # the tree positions of all the edges (we use the bottom node position)
        positions_edges = hierarchy.treepositions()[1:]  # the first one is the origin

        # map from position tuples to leaf/edge indices
        index_map_leaves = {positions_leaves[i]: i for i in range(len(positions_leaves))}
        index_map_edges = {positions_edges[i]: i for i in range(len(positions_edges))}

        # edge indices corresponding to the path from each index to the root
        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), 0, -1)] for position in positions_leaves]

        # get max size for the number of edges to the root
        num_edges = max([len(p) for p in edges_from_leaf])

        # helper that returns all leaf positions from another position wrt to the original position
        def get_leaf_positions(position, rm):
            node = hierarchy[position]
            if isinstance(node, Tree):
                if rm==0:
                    return node.treepositions("leaves")
                else:
                    b=node.treepositions("leaves")
                    return list(set([nod[:(num_edges-len(position)-rm)] for nod in b]))
            else:
                return [()]

        # indices of all leaf nodes for each edge index
        leaf_indices = [[index_map_leaves[position + leaf] for leaf in get_leaf_positions(position, rm=0)] for position in positions_edges]

        # to get the num and deno for all levels
        self.onehot_den_levels=[]
        self.onehot_num_levels=[]
        self.weights_levels=[]

        def get_numo_deno(leaf_indices, positions_edges, edges_from_leaf, num_classes, num_edges ):

            num_edges = max([len(p) for p in edges_from_leaf])

            onehot_den = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
            onehot_num = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
            whts = torch.nn.Parameter(torch.zeros([num_classes, num_edges]), requires_grad=False)

            for i in range(num_classes):
                for j, k in enumerate(edges_from_leaf[i]):
                    onehot_num[i, leaf_indices[k], j] = 1.0
                    whts[i, j] = get_label(weights[positions_edges[k]])
                for j, k in enumerate(edges_from_leaf[i][1:]):
                    onehot_den[i, leaf_indices[k], j] = 1.0
                if num_edges!=1:
                    onehot_den[i, :, j + 1] = 1.0  # the last denominator is the sum of all leaves 
                else:
                    onehot_den[i, :, 0] = 1.0      # the last denominator is the sum of all leaves (for topmost level)
            onehot_den=onehot_den.cuda()
            onehot_num=onehot_num.cuda()
            whts=whts.cuda()

            return onehot_den, onehot_num, whts

        x,y,z=get_numo_deno(leaf_indices, positions_edges, edges_from_leaf, num_classes, num_edges)
        self.onehot_den_levels.append(x)
        self.onehot_num_levels.append(y)
        self.weights_levels.append(z)

        for level in range(num_edges-1,0,-1): # from first non-leaf level to topmost level

            positions_leaves_new=list(set([  pos[:level]  for pos in positions_leaves]))
            positions_edges_new=[ed for ed in positions_edges if len(ed) <=level]
            index_map_leaves_new = {positions_leaves_new[i]: i for i in range(len(positions_leaves_new))}
            index_map_edges_new = {positions_edges_new[i]: i for i in range(len(positions_edges_new))}

            edges_from_leaf_new = [[index_map_edges_new[position[:i]] for i in range(len(position), 0, -1)] for position in positions_leaves_new]

            # get max size for the number of edges to the root
            num_edges_new = max([len(p) for p in edges_from_leaf_new])

            leaf_indices_new = [[index_map_leaves_new[position + leaf] for leaf in get_leaf_positions(position, num_edges-num_edges_new)] for position in positions_edges_new]
            x,y,z=get_numo_deno(leaf_indices_new, positions_edges_new, edges_from_leaf_new, len(positions_leaves_new), num_edges)
            self.onehot_den_levels.append(x)
            self.onehot_num_levels.append(y)
            self.weights_levels.append(z)

    def forward(self, inputs, target, level):
        """
        Foward pass, computing the loss.

        Args:
            inputs: Class _probabilities_ ordered as the input hierarchy.
            target: The index of the ground truth class.
        """

        # add a sweet dimension to inputs
        inputs = torch.unsqueeze(inputs, 1)

        # sum of probabilities for numerators
        num = torch.squeeze(torch.bmm(inputs, self.onehot_num_levels[level][target]))
        # sum of probabilities for denominators
        den = torch.squeeze(torch.bmm(inputs, self.onehot_den_levels[level][target]))
        # compute the neg logs for non zero numerators and store in there
        idx = num != 0
        num[idx] = -torch.log(num[idx] / den[idx])

        if level!=6: # we don't need to do his operation for topmost level i.e., for level==6
            # weighted sum of all logs for each path (we flip because it is numerically more stable)
            num = torch.sum(torch.flip(self.weights_levels[level][target] * num, dims=[1]), dim=1)
        else:
            num=self.weights_levels[level][target] * num
        # return sum of losses / batch size
        return torch.mean(num)


class HierarchicalCrossEntropyLoss_red_chilli(HierarchicalLLLoss_red_chilli):

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree, coeffs=(1.0,)):
        super(HierarchicalCrossEntropyLoss_red_chilli, self).__init__(hierarchy, classes, weights)

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
                super(HierarchicalCrossEntropyLoss_red_chilli, self).forward(torch.nn.functional.softmax(input, 1), target, level)
                for level, (input, target) in enumerate(zip(inputs, targets))
            ]
        )

        return torch.dot(self.coeffs, losses)


