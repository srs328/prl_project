"""Train MONAI Auto3DSeg model.

Thin wrapper that delegates to Experiment.train(). Supports both the new
OOP workflow and backward-compatible --run-dir usage.

Usage:
    # New way (preferred):
    prl train roi_train2

    # Old way (still works, used by launch_experiments.py and HPC scripts):
    python train.py --run-dir /path/to/run_dir
"""

import sys
import argparse
import logging
from pathlib import Path

from monai.config import print_config

print_config()
logging.getLogger("monai.apps.auto3dseg").setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description="Train MONAI Auto3DSeg model")
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Path to run directory. If not provided, auto-increments to next available run<N>",
    )
    parser.add_argument(
        "--dataset", type=str, default="roi_train2_t1",
        help="Dataset name (default: roi_train2_t1)",
    )
    args = parser.parse_args()

    from core.dataset import Dataset
    from core.experiment import Experiment

    ds = Dataset(args.dataset)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        exp = Experiment.from_run_dir(run_dir, ds)
    else:
        exp = Experiment(ds, ds.default_preprocess, ds.default_training,
                         run_dir=Path("."))
        exp.run_dir = exp.next_run_dir()
        exp.setup()

    exp.train()


if __name__ == "__main__":
    main()
