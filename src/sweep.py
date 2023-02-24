"""
Does a (seeded) qausi-random sweep over hyperparameters and generates config 
files that should not be committed to version control.
"""
import argparse
import pathlib

import preface
import yaml
from tqdm.auto import tqdm

import wandb

from . import config, halton


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        required=True,
        help="Base config with options that are constant across all runs.",
    )
    parser.add_argument("--sweep", required=True, help="Config with sweep options.")
    parser.add_argument(
        "--count", type=int, default=100, help="Number of trials to sample."
    )
    parser.add_argument("--output", required=True, help="Output directory.")

    return parser.parse_args()


def load(file):
    with open(file, "r") as f:
        dct = yaml.load(f, Loader=yaml.FullLoader)
    sweep_cfg = dct.pop("SWEEP")
    search_space = flatten(dct)
    return search_space, sweep_cfg


_Primitive = str | int | float | bool | None


def flatten(dct: dict[str, object], sep=".") -> dict[str, _Primitive]:
    new = {}
    for key, value in dct.items():
        # Only flatten items that have UPPERCASE keys
        if isinstance(value, dict) and all(k.upper() == k for k in value.keys()):
            for nested_key, nested_value in flatten(value).items():
                new[key + sep + nested_key] = nested_value
            continue

        new[key] = value

    return new


def main():
    args = parse_args()

    cfg = config.get_default_config()
    config.update_config_from_file(cfg, args.base)

    # Save original experiment name for later
    original_experiment_name = cfg.EXPERIMENT.NAME

    search_space, sweep_cfg = load(args.sweep)

    # Set values that are true for all trials
    cfg.defrost()

    cfg.EXPERIMENT.GOAL = sweep_cfg["GOAL"]
    cfg.EXPERIMENT.VERSION = sweep_cfg["VERSION"]

    tag = f"sweep.{original_experiment_name}-v{cfg.EXPERIMENT.VERSION}"
    cfg.EXPERIMENT.TAGS.append(tag)

    cfg.freeze()

    # Make the output directory for the generated configs
    output_dir = pathlib.Path(args.output) / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, trial in enumerate(tqdm(halton.generate_search(search_space, args.count))):
        # Make a list of [key, value, key2, value2, ...]
        trial_list = preface.flattened(list(trial.items()))

        cfg.merge_from_list(trial_list)

        # Set trial-specific values
        cfg.defrost()
        cfg.EXPERIMENT.WANDB_ID = wandb.util.generate_id()
        cfg.EXPERIMENT.NAME = f"{original_experiment_name}-{i}"
        cfg.freeze()

        # Write the file
        cfg_path = output_dir / f"{cfg.EXPERIMENT.NAME}.yaml"
        with open(cfg_path, "w") as fd:
            fd.write(cfg.dump())


if __name__ == "__main__":
    main()
