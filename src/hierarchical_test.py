""""
Uses reference implementation from the MBM paper to test a simpler implementation.
"""

import itertools
import pickle

import hypothesis
import hypothesis.extra.numpy
import numpy as np
import torch
from nltk.tree import Tree

from . import hierarchical, utils

n_classes = 8


def inat21_hierarchy():
    with open("inat21_tree.pkl", "rb") as fd:
        return pickle.load(fd)


def example_hierarchy():
    """
    Returns a binary tree of letters, from A to O.

                            A
                 ___________|_______________
                B                           C
         _______|_______             _______|_______
        D               E           F               G
     ___|___         ___|___     ___|___         ___|___
    H       I       J       K   L       M       N       O
    """

    return Tree(
        "A",
        [
            Tree(
                "B",
                [
                    Tree("D", ["H", "I"]),
                    Tree("E", ["J", "K"]),
                ],
            ),
            Tree(
                "C",
                [
                    Tree("F", ["L", "M"]),
                    Tree("G", ["N", "O"]),
                ],
            ),
        ],
    )


def example_complex_hierarchy():
    """
    Returns an irregular tree of letters.

    Things I do to make it tricky:
    * Some parents only have one child
    * Some parents have more than two children.

                    A
                 ___|_______________
                B                   C
     ___________|_______            |
    D       E           F           G
    |    ___|___     ___|___     ___|___
    H   I       J   K       L   M   N   O
    """

    return Tree(
        "A",
        [
            Tree(
                "B",
                [
                    Tree("D", ["H"]),
                    Tree("E", ["I", "J"]),
                    Tree("F", ["K", "L"]),
                ],
            ),
            Tree(
                "C",
                [
                    Tree("G", ["M", "N", "O"]),
                ],
            ),
        ],
    )


def test_mbm_smoke():
    hierarchy = example_hierarchy()
    weights = utils.get_uniform_weighting(hierarchy, 0.1)
    classes = hierarchy.leaves()

    hierarchical_nll_mbm = hierarchical.HierarchicalNLLLossMBM(
        hierarchy, classes, weights
    )


def test_mbm_smoke_complex():
    hierarchy = example_complex_hierarchy()
    weights = utils.get_uniform_weighting(hierarchy, 0.1)
    classes = hierarchy.leaves()

    hierarchical_nll_mbm = hierarchical.HierarchicalNLLLossMBM(
        hierarchy, classes, weights
    )


@hypothesis.given(
    hypothesis.extra.numpy.arrays(
        float, shape=(n_classes,), elements=hypothesis.strategies.floats(-1e1, 1e1)
    ),
    hypothesis.strategies.integers(0, n_classes - 1),
)
@hypothesis.example(logits=[0, 0, 0, 0, 0, 0, 0, 1], target_class=0)
@hypothesis.example(logits=[9, 0, 0, 0, 0, 0, 0, 0], target_class=0)
def test_mbm_loss_like_cross_entropy(logits, target_class):
    hierarchy = example_hierarchy()
    weights = utils.get_uniform_weighting(hierarchy, 1.0)
    classes = hierarchy.leaves()

    hierarchical_nll_mbm = hierarchical.HierarchicalNLLLossMBM(
        hierarchy, classes, weights
    )

    probs = torch.nn.functional.softmax(
        torch.tensor(logits, dtype=torch.float), dim=0
    ).view(1, n_classes)
    target = torch.tensor([target_class], dtype=torch.long).view(1)
    hxe_loss = hierarchical_nll_mbm(probs, target)

    # Since we use uniform weights of 1, the loss should be the same as cross entropy.
    xe_loss = torch.nn.functional.nll_loss(torch.log(probs), target)

    assert hxe_loss.shape == xe_loss.shape
    assert torch.allclose(xe_loss, hxe_loss, atol=1e-6)


@hypothesis.given(
    hypothesis.extra.numpy.arrays(
        float, shape=(n_classes,), elements=hypothesis.strategies.floats(-1e1, 1e1)
    ),
    hypothesis.strategies.integers(0, n_classes - 1),
)
def test_mbm_loss_complex_like_cross_entropy(logits, target_class):
    hierarchy = example_complex_hierarchy()
    weights = utils.get_uniform_weighting(hierarchy, 1.0)
    classes = hierarchy.leaves()

    hierarchical_nll_mbm = hierarchical.HierarchicalNLLLossMBM(
        hierarchy, classes, weights
    )

    probs = torch.nn.functional.softmax(
        torch.tensor(logits, dtype=torch.float), dim=0
    ).view(1, n_classes)
    target = torch.tensor([target_class], dtype=torch.long).view(1)
    hxe_loss = hierarchical_nll_mbm(probs, target)

    # Since we use uniform weights of 1, the loss should be the same as cross entropy.
    xe_loss = torch.nn.functional.nll_loss(torch.log(probs), target)

    assert hxe_loss.shape == xe_loss.shape
    assert torch.allclose(xe_loss, hxe_loss, atol=1e-6)


def test_mbm_loss_exponential():
    hierarchy = example_hierarchy()
    weights = utils.get_exponential_weighting(hierarchy, 0.1)
    classes = hierarchy.leaves()

    hierarchical_nll_mbm = hierarchical.HierarchicalNLLLossMBM(
        hierarchy, classes, weights
    )

    probs = torch.tensor([1.0 / n_classes] * n_classes, dtype=torch.float).view(
        1, n_classes
    )
    target_class = 0
    target = torch.tensor([target_class], dtype=torch.long).view(1)
    hxe_loss = hierarchical_nll_mbm(probs, target)

    # Manual calculations
    expected_loss = -(0.061677575 + 0.0681642630 + 0.07533316) * torch.log(
        torch.tensor(0.5)
    )

    assert torch.allclose(hxe_loss, expected_loss)


@hypothesis.given(
    hypothesis.extra.numpy.arrays(
        float, shape=(n_classes,), elements=hypothesis.strategies.floats(-1e1, 1e1)
    ),
    hypothesis.strategies.integers(0, n_classes - 1),
)
def test_efficient_loss(logits, target_cls):
    hierarchy = example_hierarchy()
    weights = utils.get_uniform_weighting(hierarchy, 1.0)
    classes = hierarchy.leaves()

    nll_mbm = hierarchical.HierarchicalNLLLossMBM(hierarchy, classes, weights)
    nll_eff = hierarchical.HierarchicalNLLLossEfficientMBM(hierarchy, classes, weights)

    probs = torch.nn.functional.softmax(
        torch.tensor(logits, dtype=torch.float), dim=0
    ).view(1, n_classes)
    target = torch.tensor([target_class], dtype=torch.long).view(1)

    mbm_loss = nll_mbm(probs, target)

    eff_loss = nll_eff(probs, target)

    assert mbm_loss.shape == eff_loss.shape
    assert torch.allclose(mbm_loss, eff_loss, atol=1e-6)


def test_onehot_num_loader():
    hierarchy = example_hierarchy()
    weights = utils.get_uniform_weighting(hierarchy, 1.0)
    classes = hierarchy.leaves()

    hierarchical_nll_mbm = hierarchical.HierarchicalNLLLossMBM(
        hierarchy, classes, weights
    )

    param_count = sum(
        [
            t.numel()
            for t in itertools.chain(
                hierarchical_nll_mbm.parameters(), hierarchical_nll_mbm.buffers()
            )
        ]
    )

    print(param_count)

    # Want to check that the one hot matrices are the same.
