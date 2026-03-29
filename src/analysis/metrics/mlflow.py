"""MLflow metric loading, aggregation, and row building.

Functions:
    load_metrics_from_file: Parse raw MLflow metric files from disk.
    analyze_unified_mlruns: Load all fold metrics from a unified mlruns directory.
    analyze_distributed_mlruns: Load fold metrics from per-fold mlruns directories.
    aggregate_metrics: Compute cross-fold statistics from fold_data.
    mlflow_metrics: Build a single summary row dict from experiment_data.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from analysis.metrics.display import rename_metric


def load_metrics_from_file(metrics_dir: Path) -> dict[str, list[tuple[float, int]]]:
    """Load metrics from an MLflow metrics directory.

    Handles both flat and nested metric directory structures.

    Returns:
        {metric_name: [(value, step), ...]} sorted by step.
    """
    metrics: dict[str, list[tuple[float, int]]] = defaultdict(list)

    if not metrics_dir.exists():
        return metrics

    for metric_file in metrics_dir.rglob("*"):
        if metric_file.is_file() and not metric_file.name.startswith("."):
            rel_path = metric_file.relative_to(metrics_dir)
            metric_name = str(rel_path)

            try:
                with open(metric_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            value = float(parts[1])
                            step = int(parts[2])
                            metrics[metric_name].append((value, step))
            except Exception as e:
                logger.warning(f"Failed to read {metric_file}: {e}")

    for metric_name in metrics:
        metrics[metric_name].sort(key=lambda x: x[1])

    return metrics


def analyze_distributed_mlruns(
    run_dir: Path, which_folds: list[int] | None = None
) -> dict[int, dict]:
    """Analyze distributed setup where each fold has its own mlruns directory."""
    fold_data = {}

    for fold_dir in sorted(run_dir.glob("segresnet_*")):
        fold_num = int(fold_dir.name.split("_")[1])
        if which_folds and fold_num not in which_folds:
            continue
        mlruns_dir = fold_dir / "model" / "mlruns"

        if not mlruns_dir.exists():
            logger.warning(f"No mlruns found in {fold_dir}")
            continue

        run_dirs = list(mlruns_dir.glob("*/*/"))
        if not run_dirs:
            logger.warning(f"No runs found in {mlruns_dir}")
            continue

        run_path = run_dirs[0]
        metrics_dir = run_path / "metrics"
        metrics = load_metrics_from_file(metrics_dir)

        fold_data[fold_num] = {
            "metrics": metrics,
            "path": str(fold_dir),
        }

    return fold_data


def analyze_unified_mlruns(run_dir: Path) -> dict[int, dict]:
    """Analyze unified setup where all folds are in a shared mlruns directory."""
    fold_data = {}
    mlruns_dir = run_dir / "mlruns"

    if not mlruns_dir.exists():
        logger.warning(f"No mlruns directory found at {mlruns_dir}")
        return fold_data

    experiment_dirs = list(mlruns_dir.glob("*"))

    for exp_dir in experiment_dirs:
        if exp_dir.name in [".trash"]:
            continue

        run_dirs = list(exp_dir.glob("*/"))

        for run_path in run_dirs:
            if run_path.name in [".trash"]:
                continue

            params_dir = run_path / "params"
            fold_num = None

            if (params_dir / "fold").exists():
                try:
                    with open(params_dir / "fold", "r") as f:
                        fold_num = int(f.read().strip())
                except Exception:
                    pass

            run_name_file = run_path / "meta.yaml"
            if run_name_file.exists() and fold_num is None:
                try:
                    import yaml

                    with open(run_name_file, "r") as f:
                        meta = yaml.safe_load(f)
                        if "run_name" in meta:
                            parts = meta["run_name"].split("fold")
                            if len(parts) > 1:
                                fold_num = int(parts[1].split()[0])
                except Exception:
                    pass

            if fold_num is None:
                logger.debug(f"Could not determine fold number for {run_path}")
                continue

            metrics_dir = run_path / "metrics"
            metrics = load_metrics_from_file(metrics_dir)

            # Handle duplicates: keep whichever has more epochs
            if fold_num in fold_data:
                old_metrics = fold_data[fold_num]["metrics"]
                try:
                    n_epochs_old = len(old_metrics["train/loss"])
                except Exception:
                    n_epochs_old = 0
                try:
                    n_epochs_new = len(metrics["train/loss"])
                except Exception:
                    continue

                if n_epochs_old > n_epochs_new:
                    continue

            fold_data[fold_num] = {
                "metrics": metrics,
                "path": str(run_path),
                "metrics_dir": str(metrics_dir),
            }

    return fold_data


def aggregate_metrics(fold_data: dict[int, dict]) -> dict[str, dict]:
    """Aggregate metrics across all folds.

    Returns:
        {metric_name: {fold_num: [values], "stats": {mean, std, min, max, best_fold_value}}}
    """
    aggregated: dict[str, dict] = defaultdict(dict)

    for fold_num in sorted(fold_data.keys()):
        metrics = fold_data[fold_num]["metrics"]
        for metric_name, values in metrics.items():
            metric_values = [v for v, _ in values]
            aggregated[metric_name][fold_num] = metric_values

    for metric_name in aggregated:
        all_values = []
        for fold_num in sorted(aggregated[metric_name].keys()):
            all_values.extend(aggregated[metric_name][fold_num])

        if all_values:
            fold_lists = [
                (fold_num, vals)
                for fold_num, vals in aggregated[metric_name].items()
                if isinstance(vals, list)
            ]
            best_fold = (
                max(
                    [(fn, max(vals)) for fn, vals in fold_lists],
                    key=lambda x: x[1],
                )
                if fold_lists
                else None
            )
            aggregated[metric_name]["stats"] = {
                "mean": np.mean(all_values),
                "std": np.std(all_values),
                "min": np.min(all_values),
                "max": np.max(all_values),
                "best_fold_value": best_fold,
            }

    return dict(aggregated)


def mlflow_metrics(experiment_data: dict) -> dict:
    """Build a single summary row dict from experiment_data.

    Flattens MLflow aggregated metrics into columns suitable for a DataFrame row.
    Includes per-fold final/max values for val_class metrics.
    """
    row = {}

    aggregated = experiment_data.get("mlflow_aggregated", {})

    if not aggregated:
        row["status"] = "incomplete"
    else:
        row["status"] = "complete"

    by_prefix: dict[str, dict] = defaultdict(dict)
    for metric_name in aggregated:
        renamed = rename_metric(metric_name)
        if "/" in renamed:
            prefix = renamed.split("/")[0]
            by_prefix[prefix][renamed] = aggregated[metric_name]
        else:
            by_prefix["other"][renamed] = aggregated[metric_name]

    fold_nums = sorted(
        k
        for k in (experiment_data.get("mlflow_fold_data") or {}).keys()
        if isinstance(k, int)
    )

    for prefix in sorted(by_prefix.keys()):
        for metric_name in sorted(by_prefix[prefix].keys()):
            metric_data = aggregated.get(
                next(
                    (
                        orig
                        for orig, data in aggregated.items()
                        if rename_metric(orig) == metric_name
                    ),
                    metric_name,
                ),
                {},
            )

            if "stats" not in metric_data:
                continue

            stats = metric_data["stats"]
            for stat in ["mean", "std", "min", "max"]:
                row[f"{metric_name}_{stat}"] = np.round(stats[stat], 4)

            if "val_class" not in metric_name and (
                "rim/acc" not in metric_name and "lesion/acc" not in metric_name
            ):
                continue
            for fold_num in fold_nums:
                if fold_num not in metric_data:
                    continue
                values = metric_data[fold_num]
                row[f"fold{fold_num}-{metric_name}_final"] = np.round(values[-1], 4)
                row[f"fold{fold_num}-{metric_name}_max"] = np.round(
                    np.max(values), 4
                )

    return row


def plot_metrics(
    fold_data: dict[int, dict],
    _aggregated: dict[str, dict] | None = None,
    output_dir: Path | None = None,
    metrics: list[str] | None = None,
):
    """Plot metrics across folds and epochs.

    For each metric, produces a two-panel figure: per-fold traces (left)
    and mean +/- std (right).

    Args:
        fold_data: Per-fold metric data from analyze_*_mlruns().
        aggregated: Output of aggregate_metrics() (unused currently, reserved).
        output_dir: If provided, save PNGs here. Otherwise display interactively.
        metrics: Subset of metric names to plot. Defaults to all common metrics.
    """
    import matplotlib.pyplot as plt

    common_metrics = set.intersection(
        *(set(fold_data[f]["metrics"].keys()) for f in fold_data)
    )
    if metrics is not None:
        common_metrics &= set(metrics)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    sorted_folds = sorted(fold_data.keys())

    for metric_name in sorted(common_metrics):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        display_name = rename_metric(metric_name)

        # Left panel: per-fold traces
        for fold_num in sorted_folds:
            values, steps = zip(*fold_data[fold_num]["metrics"][metric_name])
            ax1.plot(steps, values, label=f"Fold {fold_num}", alpha=0.7)
        ax1.set(xlabel="Step", ylabel="Value", title=f"{display_name} — All Folds")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right panel: mean +/- std aligned to common steps
        all_steps = sorted({
            step for f in fold_data
            for _, step in fold_data[f]["metrics"][metric_name]
        })
        fold_values_by_step = {step: [] for step in all_steps}
        for fold_num in sorted_folds:
            step_to_val = {step: val for val, step in fold_data[fold_num]["metrics"][metric_name]}
            for step in all_steps:
                if step in step_to_val:
                    fold_values_by_step[step].append(step_to_val[step])

        means = np.array([np.mean(fold_values_by_step[s]) if fold_values_by_step[s] else np.nan for s in all_steps])
        stds = np.array([np.std(fold_values_by_step[s]) if fold_values_by_step[s] else 0 for s in all_steps])

        ax2.plot(all_steps, means, color="black", linewidth=2, label="Mean")
        ax2.fill_between(all_steps, means - stds, means + stds, alpha=0.3, label="±1 Std")
        ax2.set(xlabel="Step", ylabel="Value", title=f"{display_name} — Mean ± Std")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir is not None:
            safe_name = metric_name.replace("/", "_")
            fig.savefig(output_dir / f"{safe_name}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()