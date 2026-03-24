"""Inspect actual runtime training parameters across runs in an experiment sweep.

Compares what was *intended* (monai_config.json) vs what *actually ran*
(training.log) for key parameters that are prone to being overridden or
swapped at runtime (batch_size, num_crops_per_image, loss config, etc.).

Also checks fold completion status via algo_object.pkl.

Usage:
    python inspect_run_params.py /path/to/experiment_dir
    python inspect_run_params.py /path/to/experiment_dir --fold 0
"""

import argparse
import json
import pickle
from pathlib import Path


RUNTIME_KEYS = [
    "batch_size",
    "num_crops_per_image",
    "num_images_per_batch",
    "num_epochs",
    "learning_rate",
    "crop_mode",
    "crop_ratios",
    "auto_scale_batch",
    "auto_scale_allowed",
]


# this will save the value of key from the last line that matches ": " in line
def parse_training_log(log_path: Path) -> dict:
    """Parse key: value lines from training.log into a dict."""
    if not log_path.exists():
        return {}
    values = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if ": " in line:
                key, _, val = line.partition(": ")
                key = key.strip()
                if key in RUNTIME_KEYS:
                    values[key] = val.strip()
    return values


def get_fold_status(fold_dir: Path) -> str:
    """Return fold completion status from algo_object.pkl."""
    pkl_path = fold_dir / "algo_object.pkl"
    if not pkl_path.exists():
        return "not started"
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    best_metric = pkl_data.get("best_metric")
    if best_metric is not None:
        return f"complete (best={best_metric:.4f})"
    progress_path = fold_dir / "model" / "progress.yaml"
    if progress_path.exists():
        return "killed (progress.yaml exists, no best_metric in pkl)"
    return "algo_gen only (never trained)"


def get_intended_params(monai_config_path: Path) -> dict:
    """Extract key params from monai_config.json train_param."""
    if not monai_config_path.exists():
        return {}
    with open(monai_config_path) as f:
        cfg = json.load(f)
    tp = cfg.get("train_param", {})
    result = {k: tp[k] for k in RUNTIME_KEYS if k in tp}
    # Summarize loss dict
    loss = tp.get("loss")
    if isinstance(loss, dict):
        result["loss.include_background"] = loss.get("include_background", "?")
        result["loss.weight"] = loss.get("weight", None)
    return result


def inspect_run(run_dir: Path, fold: int = 0) -> None:
    fold_dir = run_dir / f"segresnet_{fold}"
    log_path = fold_dir / "model" / "training.log"
    monai_config_path = run_dir / "monai_config.json"

    intended = get_intended_params(monai_config_path)
    actual = parse_training_log(log_path)
    status = get_fold_status(fold_dir) if fold_dir.exists() else "not started"

    print(f"\n{'─'*60}")
    print(f"  {run_dir.name}  /  fold {fold}  —  {status}")
    print(f"{'─'*60}")

    all_keys = sorted(set(list(intended.keys()) + list(actual.keys())))
    mismatches = []
    for key in all_keys:
        iv = str(intended.get(key, "—"))
        av = str(actual.get(key, "—"))
        flag = "  ⚠" if iv != "—" and av != "—" and iv != av else ""
        if flag:
            mismatches.append(key)
        print(f"  {key:<35} intended={iv:<15} actual={av}{flag}")

    if mismatches:
        print(f"\n  ⚠  Mismatch on: {', '.join(mismatches)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path, help="Path to experiment sweep directory (contains run1, run2, ...)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to inspect (default: 0)")
    parser.add_argument("--all-folds", action="store_true", help="Inspect all 5 folds per run")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.exists():
        print(f"Error: {experiment_dir} does not exist")
        return

    run_dirs = sorted(
        [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("run")],
        key=lambda d: d.name,
    )

    if not run_dirs:
        print(f"No run* directories found in {experiment_dir}")
        return

    print(f"\nExperiment: {experiment_dir}")
    print(f"Runs found: {[d.name for d in run_dirs]}")

    folds = list(range(5)) if args.all_folds else [args.fold]

    for run_dir in run_dirs:
        for fold in folds:
            inspect_run(run_dir, fold)


if __name__ == "__main__":
    main()
