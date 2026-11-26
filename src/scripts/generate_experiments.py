#!/usr/bin/env python
"""
Generate HPO experiment configs from a parameter grid.

Reads an experiment_config.json, computes the Cartesian product of parameter combinations,
and generates individual run directories with customized label_config.json and monai_config.json.
Also pre-generates stacked images for each configuration.
"""

import sys
import os
import json
import argparse
import shutil
from pathlib import Path
from itertools import product
import subprocess
from copy import deepcopy

from helpers.paths import load_config, PROJECT_ROOT


def set_nested_value(obj, key_path, value):
    """
    Set a value in a nested dict using dot notation (e.g., 'train_param.learning_rate').
    """
    keys = key_path.split('.')
    current = obj
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_nested_value(obj, key_path):
    """Get a value in a nested dict using dot notation."""
    keys = key_path.split('.')
    current = obj
    for key in keys:
        current = current[key]
    return current


def delete_nested_key(obj, key_path):
    """
    Delete a key from a nested dict using dot notation.
    If the key doesn't exist, do nothing.
    """
    keys = key_path.split('.')
    current = obj
    for key in keys[:-1]:
        if key not in current:
            return
        current = current[key]
    current.pop(keys[-1], None)


def make_argument_parser(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        description="Generate HPO experiment configs from parameter grid"
    )
    parser.add_argument("experiment_config", type=str, help="Path to experiment_config.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be generated without writing files"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        default=False,
        help="Run prepare_training_data.py for each generated config"
    )
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = make_argument_parser(argv)
    args = parser.parse_args()


    # Load experiment config
    exp_config = load_config(args.experiment_config)
    train_home = Path(exp_config['train_home'])
    exp_name = exp_config.get("experiment_name", "hpo_experiment")
    param_grid = exp_config["param_grid"]

    # Load base configs
    base_label_config = load_config(train_home/exp_config["base_label_config"])
    base_monai_config = load_config(train_home/exp_config["base_monai_config"])

    # Put experiment runs in a subdirectory named after the experiment
    training_work_home = Path(base_monai_config["training_work_home"]) / exp_name

    # Build parameter combinations
    label_params = param_grid.get("label", {})
    monai_params = param_grid.get("monai", {})

    # Create all combinations
    label_param_keys = list(label_params.keys())
    label_param_values = [label_params[k] for k in label_param_keys]

    monai_param_keys = list(monai_params.keys())
    monai_param_values = [monai_params[k] for k in monai_param_keys]

    label_combinations = list(product(*label_param_values)) if label_param_keys else [()]
    monai_combinations = list(product(*monai_param_values)) if monai_param_keys else [()]

    # Generate all runs
    runs_manifest = {}
    run_num = 1

    for label_combo in label_combinations:
        for monai_combo in monai_combinations:
            run_dir = training_work_home / f"run{run_num}"
            run_name = f"run{run_num}"

            # Build param dicts for this combination
            label_params_dict = dict(zip(label_param_keys, label_combo))
            monai_params_dict = dict(zip(monai_param_keys, monai_combo))

            # Create customized configs
            label_cfg = deepcopy(base_label_config)
            for param_key, param_val in label_params_dict.items():
                if param_val is None:
                    label_cfg.pop(param_key, None)
                else:
                    label_cfg[param_key] = param_val

            monai_cfg = deepcopy(base_monai_config)
            # Point training_work_home to the experiment subdirectory
            monai_cfg["training_work_home"] = str(training_work_home)
            for param_key, param_val in monai_params_dict.items():
                if param_val is None:
                    # Remove the key so the downstream default is used
                    delete_nested_key(monai_cfg, param_key)
                else:
                    set_nested_value(monai_cfg, param_key, param_val)

            # Record in manifest
            runs_manifest[run_name] = {
                "label_params": label_params_dict,
                "monai_params": monai_params_dict,
                "run_dir": str(run_dir),
            }

            if args.dry_run:
                print(f"\n[DRY RUN] {run_name}:")
                print(f"  Label params: {label_params_dict}")
                print(f"  Monai params: {monai_params_dict}")
                print(f"  Run dir: {run_dir}")
            else:
                # Create run directory
                run_dir.mkdir(parents=True, exist_ok=True)

                # Write configs
                with open(run_dir / "label_config.json", "w") as f:
                    json.dump(label_cfg, f, indent=2)

                with open(run_dir / "monai_config.json", "w") as f:
                    json.dump(monai_cfg, f, indent=2)

                # Write params.json for reference
                with open(run_dir / "params.json", "w") as f:
                    json.dump(
                        {
                            "label_params": label_params_dict,
                            "monai_params": monai_params_dict,
                        },
                        f,
                        indent=2,
                    )

                print(f"Generated {run_name} in {run_dir}")

                # Optionally prepare data
                if args.prepare_data:
                    print(f"  Preparing training data...")
                    prepare_cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "src/preprocessing" / "prepare_training_data.py"),
                        str(run_dir / "label_config.json"),
                    ]
                    try:
                        subprocess.run(prepare_cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"  ERROR: prepare_training_data.py failed for {run_name}")
                        print(f"  {e}")

            run_num += 1

    # Write manifest
    if not args.dry_run:
        manifest_path = training_work_home / "runs_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(runs_manifest, f, indent=2)
        print(f"\nManifest saved to {manifest_path}")
        print(f"Total runs: {len(runs_manifest)}")
    else:
        print(f"\n[DRY RUN] Would generate {run_num - 1} total runs")


if __name__ == "__main__":
    main()
