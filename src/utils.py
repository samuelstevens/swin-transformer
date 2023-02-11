# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import dataclasses
import os

import torch
import torch.distributed as dist
from torch._six import inf

import pickle
import lzma
import numpy as np
from math import exp, fsum
from nltk.tree import Tree
from copy import deepcopy

@dataclasses.dataclass(frozen=True)
class ModelCheckpoint:
    max_accuracy: float
    start_epoch: int
    state_dict_msg: object


def load_model_checkpoint(checkpoint_file, model, logger) -> ModelCheckpoint:
    logger.info("Loading model checkpoint. [path: %s]", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    msg = model.load_state_dict(checkpoint["model"], strict=False)

    max_accuracy = 0.0
    if "max_accuracy" in checkpoint:
        max_accuracy = checkpoint["max_accuracy"]

    logger.info(
        "Loaded model checkpoint. [path: %s, epoch: %d]",
        checkpoint_file,
        checkpoint["epoch"],
    )

    return ModelCheckpoint(max_accuracy, checkpoint["epoch"] + 1, msg)


def load_training_checkpoint(
    checkpoint_file, optimizer, lr_scheduler, loss_scaler, logger
) -> None:
    logger.info("Loading training checkpoint. [path: %s]", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if "scaler" in checkpoint:
        loss_scaler.load_state_dict(checkpoint["scaler"])

    logger.info(
        "Loaded training checkpoint. [path: %s, epoch: %d]",
        checkpoint_file,
        checkpoint["epoch"],
    )

    del checkpoint
    torch.cuda.empty_cache()


def _hierarchical_weight_k(i):
    return f"head.heads.{i}.weight"


def _hierarchical_bias_k(i):
    return f"head.heads.{i}.bias"


def map_linear_head(state_dict, *, map_path, pt_head: str, c_head: str):
    with open(map_path) as f:
        map_indices = [int(id22k.strip()) for id22k in f.readlines()]

    state_dict[f"{c_head}.weight"] = state_dict[f"{pt_head}.weight"][map_indices, :]
    state_dict[f"{c_head}.bias"] = state_dict[f"{pt_head}.bias"][map_indices]


def handle_linear_head(config, model, state_dict, logger) -> set[str]:
    """
    Check classifier, if not match, then re-init classifier to zero
    Ways it could not match:

    1. Both have a single linear head, with different number of classes

    2. Pretrained with hierarchical head and are now finetuning with single linear head.
       If the fine-grained head was 10K classes and the linear head has 10K classes,
       and the current dataset is iNat, we use the pre-trained fine-grained head.

    3. Pretrained with single linear head and are finetuning with hierarchical head
       (for instance, if starting with imagenet pre-training, then doing domain-specific
       pre-training on iNat21)

    4. Both have a hierarchical head with different number of tiers/classes.
       We always reinitialize the hierarchical head.
    """

    pretrained_hierarchical = _hierarchical_bias_k(0) in state_dict
    current_hierarchical = config.HIERARCHICAL
    okay_missing_keys = set()

    if not pretrained_hierarchical and not current_hierarchical:
        # TESTED because Microsoft wrote this code.
        # Both have a single linear head
        assert "head.bias" in state_dict, "Should have a single pre-trained linear head"
        assert hasattr(model.head, "bias"), "Should have a single random linear head"

        head_bias_pretrained = state_dict["head.bias"]
        num_classes_pretrained = head_bias_pretrained.shape[0]
        num_classes = model.head.bias.shape[0]
        if num_classes_pretrained == num_classes:
            pass  # Don't need to do anything
        elif config.MODEL.LINEAR_HEAD_MAP_FILE:
            map_linear_head(
                state_dict,
                map_path=config.MODEL.LINEAR_HEAD_MAP_FILE,
                pt_head="head",
                c_head="head",
            )
        else:
            del state_dict["head.weight"]
            del state_dict["head.bias"]
            # Re-inits the linear layer
            model.head.reset_parameters()

            logger.warning(
                "Error in loading classifier head, randomly re-init classifier head."
            )
            okay_missing_keys.update(['head.bias', 'head.weight'])
    elif pretrained_hierarchical and not current_hierarchical:
        assert (
            "head.bias" not in state_dict
        ), "Should not have a single pre-trained linear head"
        assert hasattr(model.head, "bias"), "Should have a single random linear head"

        # Going to try to use the pretrained fine-grained linear head as the
        # initialization for the current model.

        # Increment finegrained level until the key doesn't exist.
        # Then it is the last level in the hierarchical model
        max_level = -1
        while _hierarchical_bias_k(max_level + 1) in state_dict:
            max_level += 1

        finegrained_num_classes_pretrained = state_dict[
            _hierarchical_bias_k(max_level)
        ].shape[0]
        num_classes = model.head.bias.shape[0]

        if finegrained_num_classes_pretrained == num_classes:
            # Direct mapping
            state_dict["head.weight"] = state_dict[_hierarchical_weight_k(max_level)]
            state_dict["head.bias"] = state_dict[_hierarchical_bias_k(max_level)]
        elif config.MODEL.LINEAR_HEAD_MAP_FILE:
            map_linear_head(
                state_dict,
                map_path=config.MODEL.LINEAR_HEAD_MAP_FILE,
                pt_head=f"head.heads.{max_level}",
                c_head="head",
            )
        else:
            okay_missing_keys = {"head.bias", "head.weight"}
            logger.warning(
                "Error in loading classifier head, using default initialization."
            )

        for i in range(max_level):
            del state_dict[_hierarchical_weight_k(i)]
            del state_dict[_hierarchical_bias_k(i)]

    elif not pretrained_hierarchical and current_hierarchical:
        # UNTESTED
        assert "head.bias" in state_dict, "Should have a single pre-trained linear head"
        assert not hasattr(
            model.head, "bias"
        ), "Should not have a single random linear head"

        # Delete the head.bias and head.weight keys then do nothing since the linear
        # layer is already correctly initialized from scratch so it can fine-tune.
        del state_dict["head.weight"]
        del state_dict["head.bias"]

    elif pretrained_hierarchical and current_hierarchical:
        assert (
            "head.bias" not in state_dict
        ), "Should not have a single pre-trained linear head"
        assert not hasattr(
            model.head, "bias"
        ), "Should not have a single random linear head"

        # Check if the two models have the exact same number of levels, and the same
        # number of classes in each level
        matches = True
        level = 0
        while matches and _hierarchical_bias_k(level) in state_dict:
            # Check that the current model has the right attribute
            if level > len(model.head.heads):
                matches = False
                continue

            if not hasattr(model.head.heads[level], "bias"):
                matches = False
                continue

            if (
                model.head.heads[level].bias.shape
                != state_dict[_hierarchical_bias_k(level)].shape
            ):
                matches = False
                continue

            if (
                model.head.heads[level].weight.shape
                != state_dict[_hierarchical_weight_k(level)].shape
            ):
                matches = False
                continue

            level += 1

        if not matches:
            # UNTESTED
            logger.warning(
                "Not using pre-trained hierarchical head because the shapes do not match."
            )
            # Delete the keys from the state dict because the pre-trained model and
            # the current model do not match in size.
            for i in range(level):
                del state_dict[_hierarchical_weight_k(i)]
                del state_dict[_hierarchical_bias_k(i)]
        else:
            logger.info("Using pre-trained hierarchical head.")

    return okay_missing_keys


def load_pretrained(config, model, logger):
    logger.info("Loading weights for fine-tuning. [path: %s]", config.MODEL.PRETRAINED)
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location="cpu")
    state_dict = checkpoint["model"]

    # delete relative_position_index since we always re-init it (it's a fixed buffer).
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_position_index" in k
    ]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it (it's a fixed buffer).
    relative_coords_table_keys = [
        k for k in state_dict.keys() if "relative_coords_table" in k
    ]
    for k in relative_coords_table_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it (also a fixed buffer).
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # These functions only change the positional parameters if there is a change
    # in model architecture (image size, patch size, etc).
    interpolate_rel_pos_bias(state_dict, model, logger)
    interpolate_abs_pos_embed(state_dict, model, logger)

    # This function handles changes in the linear head (from hierarchical pretraining
    # to traditional classification, for instance.
    okay_missing_head_keys = handle_linear_head(config, model, state_dict, logger)

    msg = model.load_state_dict(state_dict, strict=False)
    for key in msg.missing_keys:
        assert (
            key in okay_missing_head_keys
            or key in relative_coords_table_keys
            or key in relative_position_index_keys
            or key in attn_mask_keys
        ), f"Should only reinitialize relative positional information, not '{key}'"
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def interpolate_rel_pos_bias(state_dict, model, logger):
    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k
    ]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        elif L1 != L2:
            # bicubic interpolate relative_position_bias_table if not match
            S1 = int(L1**0.5)
            S2 = int(L2**0.5)
            relative_position_bias_table_pretrained_resized = (
                torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(
                        1, nH1, S1, S1
                    ),
                    size=(S2, S2),
                    mode="bicubic",
                )
            )
            state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                nH2, L2
            ).permute(1, 0)


def interpolate_abs_pos_embed(state_dict, model, logger):
    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if "absolute_pos_embed" in k
    ]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        elif L1 != L2:
            S1 = int(L1**0.5)
            S2 = int(L2**0.5)
            absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                -1, S1, S1, C1
            )
            absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                0, 3, 1, 2
            )
            absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                absolute_pos_embed_pretrained, size=(S2, S2), mode="bicubic"
            )
            absolute_pos_embed_pretrained_resized = (
                absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
            )
            absolute_pos_embed_pretrained_resized = (
                absolute_pos_embed_pretrained_resized.flatten(1, 2)
            )
            state_dict[k] = absolute_pos_embed_pretrained_resized


def save_checkpoint(
    config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "scaler": loss_scaler.state_dict(),
        "epoch": epoch,
        "config": config,
    }

    save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    print(f"All checkpoints found in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def batch_size_of(tensor_or_list):
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.size(0)
    elif isinstance(tensor_or_list, list):
        sizes = [tensor.size(0) for tensor in tensor_or_list]
        assert all(size == sizes[0] for size in sizes)
        return sizes[0]


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


############################### Added below script to generate .yaml files  ###########################


from typing import Any, Dict, Iterator, List, Mapping, Tuple
from . import config

def files_with_extension(paths: List[str], ext: str) -> Iterator[str]:
    # add . to ext if it doesn't have it.
    ext = "." + ext if ext[0] != "." else ext

    for path in paths:
        if os.path.isfile(path):
            if path.endswith(ext):
                yield path
                continue

        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(ext):
                    yield os.path.join(dirpath, filename) 

def _flatten_dict_of_lists(
    dict_of_lists: Mapping[object, List[object]],
) -> Iterator[Dict[object, object]]:

    assert isinstance(dict_of_lists, dict)

    if not dict_of_lists:
        yield dict_of_lists
        return

    field, value_list = dict_of_lists.popitem()

    assert isinstance(value_list, list)

    for value in value_list:
        for flattened_dict in _flatten_dict_of_lists(dict_of_lists):
            yield {**flattened_dict, field: value}
    dict_of_lists[field] = value_list

def flattened(obj: object) -> Iterator[Any]:
    """
    Given a list, dictionary, or other value, yields an iterator of dictionaries/other values with no lists anywhere inside the structure.
    """
    if not isinstance(obj, dict) and not isinstance(obj, list):
        yield obj
        return

    if isinstance(obj, list):
        for elem in obj:
            yield from flattened(elem)
        return

    assert isinstance(obj, dict)
    
    if not obj:
        yield obj
        return
    flat_list = {field: list(flattened(value)) for field, value in obj.items()}
    yield from _flatten_dict_of_lists(flat_list)


def find_experiments(paths): # -> Iterator[config.ExperimentConfig]:
    """
    Arguments:
    * args (list[str]): list of strings that are either directories containing files or config files themselves.
    """
    if not isinstance(paths, list):
        paths = [paths]

    for config_file in files_with_extension(paths, ".yaml"):
        yield config_file



# Added to incorporate the new all-level hierarchical loss (paper: makes better mistakes)

def get_label(node):
    if isinstance(node, Tree):
        return node.label()
    else:
        return node

def load_hierarchy(dataset, data_dir):
    """
    Load the hierarchy corresponding to a given dataset.
    Args:
        dataset: The dataset name for which the hierarchy should be loaded.
        data_dir: The directory where the hierarchy files are stored.
    Returns:
        A nltk tree whose labels corresponds to wordnet wnids.
    """
    if dataset in ["inaturalist21-192", "inaturalist21-224"]:
        fname = os.path.join(data_dir, "inat21_tree.pkl")
    # if dataset in ["tiered-imagenet-84", "tiered-imagenet-224"]:
    #     fname = os.path.join(data_dir, "tiered_imagenet_tree.pkl")
    # elif dataset in ["ilsvrc12", "imagenet"]:
    #     fname = os.path.join(data_dir, "imagenet_tree.pkl")
    # elif dataset in ["inaturalist19-84", "inaturalist19-224"]:
    #     fname = os.path.join(data_dir, "inaturalist19_tree.pkl")
    else:
        raise ValueError("Unknown dataset {}".format(dataset))

    with open(fname, "rb") as f:
        return pickle.load(f)

def get_uniform_weighting(hierarchy: Tree, value):
    """
    Construct unit weighting tree from hierarchy.

    Args:
        hierarchy: The hierarchy to use to generate the weights.
        value: The value to fill the tree with.

    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    for p in weights.treepositions():
        node = weights[p]
        if isinstance(node, Tree):
            node.set_label(value)
        else:
            weights[p] = value
    return weights


def get_exponential_weighting(hierarchy: Tree, value, normalize=True):
    """
    Construct exponentially decreasing weighting, where each edge is weighted
    according to its distance from the root as exp(-value*dist).

    Args:
        hierarchy: The hierarchy to use to generate the weights.
        value: The decay value.
        normalize: If True ensures that the sum of all weights sums
            to one.

    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    all_weights = []
    for p in weights.treepositions():
        node = weights[p]
        weight = exp(-value * len(p))
        all_weights.append(weight)
        if isinstance(node, Tree):
            node.set_label(weight)
        else:
            weights[p] = weight
    total = fsum(all_weights)  # stable sum
    if normalize:
        for p in weights.treepositions():
            node = weights[p]
            if isinstance(node, Tree):
                node.set_label(node.label() / total)
            else:
                weights[p] /= total
    return weights


def get_weighting(hierarchy: Tree, weighting, value):
    """
    Get different weightings of edges in a tree.

    Args:
        hierarchy: The tree to generate the weighting for.
        weighting: The type of weighting, one of 'uniform', 'exponential'.
        **kwards: Keyword arguments passed to the weighting function.
    """
    if weighting == "uniform":
        return get_uniform_weighting(hierarchy, value)
    elif weighting == "exponential":
        return get_exponential_weighting(hierarchy, value)
    else:
        raise NotImplementedError("Weighting {} is not implemented".format(weighting))


