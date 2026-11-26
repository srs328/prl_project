#!/usr/bin/env python3
"""
Analyze MLflow runs and aggregate metrics across folds.

Supports both:
- Distributed setup (each fold has its own mlruns directory)
- Unified setup (all folds logged to shared mlruns directory)

Usage:
    # Distributed setup (run2)
    python analyze_mlflow_runs.py /path/to/run2 --distributed

    # Unified setup (future runs)
    python analyze_mlflow_runs.py /path/to/run3

    # Unified setup with plots
    python analyze_mlflow_runs.py /path/to/run3 --plot
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_metrics_from_file(metrics_dir: Path) -> Dict[str, List[Tuple[float, int]]]:
    """Load metrics from MLflow metrics directory.

    Handles both flat and nested metric directory structures.
    Returns dict of {metric_name: [(value, step), ...]}
    """
    metrics = defaultdict(list)

    if not metrics_dir.exists():
        return metrics

    # Recursively find all metric files
    for metric_file in metrics_dir.rglob("*"):
        if metric_file.is_file() and not metric_file.name.startswith("."):
            # Build metric name from relative path
            rel_path = metric_file.relative_to(metrics_dir)
            metric_name = str(rel_path)

            try:
                with open(metric_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        # MLflow format: timestamp value step
                        if len(parts) >= 3:
                            value = float(parts[1])
                            step = int(parts[2])
                            metrics[metric_name].append((value, step))
            except Exception as e:
                print(f"Warning: Failed to read {metric_file}: {e}")

    # Sort by step
    for metric_name in metrics:
        metrics[metric_name].sort(key=lambda x: x[1])

    return metrics


def analyze_distributed_mlruns(run_dir: Path, which_folds=None) -> Dict[int, Dict]:
    """Analyze distributed setup where each fold has its own mlruns directory."""
    fold_data = {}

    # Find all segresnet_<fold> directories
    for fold_dir in sorted(run_dir.glob("segresnet_*")):
        fold_num = int(fold_dir.name.split("_")[1])
        if which_folds and fold_num not in which_folds:
            continue
        mlruns_dir = fold_dir / "model" / "mlruns"

        if not mlruns_dir.exists():
            print(f"Warning: No mlruns found in {fold_dir}")
            continue

        # Find the run directory (usually under experiment_id/run_id)
        run_dirs = list(mlruns_dir.glob("*/*/"))
        if not run_dirs:
            print(f"Warning: No runs found in {mlruns_dir}")
            continue

        # Use the first (and usually only) run
        run_path = run_dirs[0]
        metrics_dir = run_path / "metrics"
        metrics = load_metrics_from_file(metrics_dir)

        fold_data[fold_num] = {
            "metrics": metrics,
            "path": str(fold_dir),
        }

    return fold_data


def analyze_unified_mlruns(run_dir: Path) -> Dict[int, Dict]:
    """Analyze unified setup where all folds are in shared mlruns directory."""
    fold_data = {}
    mlruns_dir = run_dir / "mlruns"

    if not mlruns_dir.exists():
        print(f"Error: No mlruns directory found at {mlruns_dir}")
        return fold_data

    # Find all runs (usually under experiment_id/run_id)
    experiment_dirs = list(mlruns_dir.glob("*"))

    for exp_dir in experiment_dirs:
        if exp_dir.name in [".trash"]:
            continue

        run_dirs = list(exp_dir.glob("*/"))

        for run_path in run_dirs:
            if run_path.name in [".trash"]:
                continue

            # Try to read params to get fold number
            params_dir = run_path / "params"
            fold_num = None

            if (params_dir / "fold").exists():
                try:
                    with open(params_dir / "fold", 'r') as f:
                        fold_num = int(f.read().strip())
                except:
                    pass

            # If no fold param found, try to extract from run name
            run_name_file = run_path / "meta.yaml"
            if run_name_file.exists() and fold_num is None:
                try:
                    import yaml
                    with open(run_name_file, 'r') as f:
                        meta = yaml.safe_load(f)
                        if "run_name" in meta:
                            # Extract fold number from run name like "segresnet - fold0 - train"
                            parts = meta["run_name"].split("fold")
                            if len(parts) > 1:
                                fold_num = int(parts[1].split()[0])
                except:
                    pass

            if fold_num is None:
                print(f"Warning: Could not determine fold number for {run_path}")
                continue

            metrics_dir = run_path / "metrics"
            metrics = load_metrics_from_file(metrics_dir)

            # Handle duplicates that occur if I had to rerun a job on the cluster that failed
            if fold_num in fold_data.keys():
                old_metrics = fold_data[fold_num]["metrics"]
                old_metric_dir = fold_data[fold_num]['metrics_dir']
                try:
                    n_epochs_old = len(old_metrics['train/loss'])
                except Exception:
                    print(f"Duplicate fold {fold_num} and problem in {old_metric_dir}")
                    n_epochs_old = 0
                try:
                    n_epochs_new = len(metrics['train/loss'])
                except Exception:
                    print(f"Duplicate fold {fold_num} and problem in {metrics_dir}")
                    continue
                
                # assume whichever one has the most epochs is the one that is correct
                if n_epochs_old > n_epochs_new:
                    continue
            
            fold_data[fold_num] = {
                "metrics": metrics,
                "path": str(run_path),
                "metrics_dir": str(metrics_dir)
            }

    return fold_data


def aggregate_metrics(fold_data: Dict[int, Dict]) -> Dict[str, Dict]:
    """Aggregate metrics across all folds.

    Returns dict of {metric_name: {fold_num: values, "stats": {...}}}
    """
    aggregated = defaultdict(lambda: {})

    # Collect all metric names and values for each fold
    for fold_num in sorted(fold_data.keys()):
        metrics = fold_data[fold_num]["metrics"]

        for metric_name, values in metrics.items():
            if metric_name not in aggregated:
                aggregated[metric_name] = {}

            # Extract just the values (ignore steps for aggregation)
            metric_values = [v for v, _ in values]
            aggregated[metric_name][fold_num] = metric_values

    # Calculate statistics
    stats = {}
    for metric_name in aggregated:
        all_values = []
        for fold_num in sorted(aggregated[metric_name].keys()):
            all_values.extend(aggregated[metric_name][fold_num])

        if all_values:
            aggregated[metric_name]["stats"] = {
                "mean": np.mean(all_values),
                "std": np.std(all_values),
                "min": np.min(all_values),
                "max": np.max(all_values),
                "best_fold_value": max(
                    [(fold_num, max(vals)) for fold_num, vals in aggregated[metric_name].items()
                     if isinstance(vals, list)],
                    key=lambda x: x[1]
                ) if any(isinstance(v, list) for v in aggregated[metric_name].values()) else None,
            }

    return aggregated


def create_summary(fold_data: Dict[int, Dict], aggregated: Dict[str, Dict], fp=None):
    pass


def print_summary(fold_data: Dict[int, Dict], aggregated: Dict[str, Dict], fp=None):
    """Print a nice summary of metrics."""
    print("\n" + "="*80, file=fp)
    print("CROSS-FOLD METRICS SUMMARY", file=fp)
    print("="*80, file=fp)

    print(f"\nFolds found: {sorted(fold_data.keys())}", file=fp)
    print("\nKey metrics to evaluate:", file=fp)
    print("  • train/loss: Should be decreasing - indicates learning is happening", file=fp)
    print("  • val/acc: Main performance metric - higher is better", file=fp)
    print("  • Fold variance: High std indicates instability across folds", file=fp)
    print("  • Per-fold comparison: Look for outlier folds (very high or low)", file=fp)
    print("", file=fp)

    # Group metrics by prefix (e.g., "train/", "val/")
    by_prefix = defaultdict(dict)
    for metric_name in aggregated:
        if "/" in metric_name:
            prefix = metric_name.split("/")[0]
            by_prefix[prefix][metric_name] = aggregated[metric_name]
        else:
            by_prefix["other"][metric_name] = aggregated[metric_name]

    for prefix in sorted(by_prefix.keys()):
        print(f"\n{prefix.upper()} Metrics:", file=fp)
        print("-" * 80, file=fp)

        for metric_name in sorted(by_prefix[prefix].keys()):
            metric_data = aggregated[metric_name]

            if "stats" not in metric_data:
                continue

            stats = metric_data["stats"]

            print(f"\n  {metric_name}:", file=fp)
            print(f"    Mean:     {stats['mean']:.6f}", file=fp)
            print(f"    Std:      {stats['std']:.6f}", file=fp)
            print(f"    Min:      {stats['min']:.6f}", file=fp)
            print(f"    Max:      {stats['max']:.6f}", file=fp)

            # Print per-fold value
            fold_values = [(fold_num, np.mean(metric_data[fold_num]))
                          for fold_num in sorted(fold_data.keys())
                          if fold_num in metric_data]
            if fold_values:
                print("    Per-fold (mean across epochs):", file=fp)
                fold_values_sorted = sorted(fold_values, key=lambda x: x[1], reverse=True)
                for fold_num, val in fold_values_sorted:
                    marker = " ← Best" if fold_num == fold_values_sorted[0][0] else \
                             " ← Worst" if fold_num == fold_values_sorted[-1][0] else ""
                    print(f"      Fold {fold_num}: {val:.6f}{marker}", file=fp)


def plot_metrics(fold_data: Dict[int, Dict], aggregated: Dict[str, Dict], output_dir: Path):
    """Create plots of metrics across folds and epochs."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get common metric names across all folds
    common_metrics = set()
    for fold_num in fold_data:
        metrics = fold_data[fold_num]["metrics"]
        if not common_metrics:
            common_metrics = set(metrics.keys())
        else:
            common_metrics = common_metrics.intersection(set(metrics.keys()))

    # Plot each metric
    for metric_name in sorted(common_metrics):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: All folds overlaid
        for fold_num in sorted(fold_data.keys()):
            values, steps = zip(*fold_data[fold_num]["metrics"][metric_name])
            ax1.plot(steps, values, label=f"Fold {fold_num}", alpha=0.7)

        ax1.set_xlabel("Step")
        ax1.set_ylabel("Value")
        ax1.set_title(f"{metric_name} - All Folds")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Mean and std across folds
        # Align metrics to common steps
        all_steps = set()
        for fold_num in fold_data:
            steps = [step for _, step in fold_data[fold_num]["metrics"][metric_name]]
            all_steps.update(steps)

        all_steps = sorted(all_steps)
        fold_values_by_step = {step: [] for step in all_steps}

        for fold_num in sorted(fold_data.keys()):
            metric_dict = {step: val for val, step in fold_data[fold_num]["metrics"][metric_name]}
            for step in all_steps:
                if step in metric_dict:
                    fold_values_by_step[step].append(metric_dict[step])

        means = [np.mean(fold_values_by_step[step]) if fold_values_by_step[step] else np.nan
                for step in all_steps]
        stds = [np.std(fold_values_by_step[step]) if fold_values_by_step[step] else 0
               for step in all_steps]

        ax2.plot(all_steps, means, color='black', linewidth=2, label='Mean')
        ax2.fill_between(all_steps,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.3, label='±1 Std')

        ax2.set_xlabel("Step")
        ax2.set_ylabel("Value")
        ax2.set_title(f"{metric_name} - Mean ± Std")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        safe_name = metric_name.replace("/", "_")
        output_path = output_dir / f"{safe_name}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        plt.close()


def interpret_results(fold_data: Dict[int, Dict], aggregated: Dict[str, Dict], fp=None):
    """Provide interpretation and recommendations."""
    print("\n" + "="*80, file=fp)
    print("INTERPRETATION & NEXT STEPS", file=fp)
    print("="*80, file=fp)

    # Find key metrics
    val_acc = aggregated.get("val/acc", {})
    train_loss = aggregated.get("train/loss", {})

    if "stats" in val_acc:
        val_stats = val_acc["stats"]
        print(f"\nValidation Accuracy:", file=fp)
        print(f"  Mean: {val_stats['mean']:.4f} ± {val_stats['std']:.4f}", file=fp)
        print(f"  Range: {val_stats['min']:.4f} - {val_stats['max']:.4f}", file=fp)

        # Variance interpretation
        cv = val_stats['std'] / (val_stats['mean'] + 1e-6)  # Coefficient of variation
        if cv > 0.2:
            print(f"  ⚠ High variance (CV={cv:.2f}): Model unstable across folds", file=fp)
            print(f"    → Check for data quality issues or fold-specific problems", file=fp)
        else:
            print(f"  ✓ Reasonable consistency across folds", file=fp)

    if "stats" in train_loss:
        loss_stats = train_loss["stats"]
        print(f"\nTraining Loss:", file=fp)
        print(f"  Mean: {loss_stats['mean']:.4f} ± {loss_stats['std']:.4f}", file=fp)
        if loss_stats['std'] > loss_stats['mean'] * 0.3:
            print(f"  ⚠ High variance: Folds converging differently", file=fp)

    print(f"\nPerformance Variation:", file=fp)
    if val_acc and "stats" in val_acc:
        best_fold = max([(k, np.mean(val_acc[k]) if isinstance(val_acc[k], list) else 0)
                        for k in val_acc if isinstance(val_acc[k], list)],
                       key=lambda x: x[1], default=(None, 0))
        worst_fold = min([(k, np.mean(val_acc[k]) if isinstance(val_acc[k], list) else 0)
                         for k in val_acc if isinstance(val_acc[k], list)],
                        key=lambda x: x[1], default=(None, 0))

        if best_fold[0] is not None and worst_fold[0] is not None:
            improvement = (best_fold[1] - worst_fold[1]) / (worst_fold[1] + 1e-6)
            print(f"  Best fold (Fold {best_fold[0]}): {best_fold[1]:.4f}", file=fp)
            print(f"  Worst fold (Fold {worst_fold[0]}): {worst_fold[1]:.4f}", file=fp)
            print(f"  Potential improvement: {improvement*100:.1f}%", file=fp)

    print(f"\nRecommendations:", file=fp)
    print(f"  1. Check outlier folds for data quality issues", file=fp)
    print(f"  2. Verify fold assignment is balanced (look at datalist)", file=fp)
    print(f"  3. Consider if high variance is expected (few samples per fold?)", file=fp)
    print(f"  4. Next: tune hyperparameters or collect more data", file=fp)


def main():
    parser = argparse.ArgumentParser(description="Analyze MLflow runs across folds")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--distributed", action="store_true",
                       help="Use distributed analysis (each fold has separate mlruns)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots (requires matplotlib)")
    parser.add_argument("--outfile", type=Path, default=None, help="Save output to file instead of printing to console")
    parser.add_argument("-f", "--fold", action="extend", default=None, help="Folds to analyze")
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}")
        return 1

    print(f"Analyzing run: {args.run_dir}")
    
    which_folds = args.fold
    if which_folds is not None:
        which_folds = [int(f) for f in which_folds]

    # Load metrics
    if args.distributed:
        print("Using distributed analysis...")
        fold_data = analyze_distributed_mlruns(args.run_dir, which_folds=which_folds)
    else:
        print("Using unified analysis...")
        fold_data = analyze_unified_mlruns(args.run_dir)

    if not fold_data:
        print("Error: No fold data found")
        return 1

    print(f"Found {len(fold_data)} folds")

    # Aggregate and display
    aggregated = aggregate_metrics(fold_data)
    output_path = args.outfile
    if output_path:
        if not output_path.is_absolute():
            output_path = args.run_dir / output_path
        with open(output_path, 'w') as f:
            print(f"To {output_path}")
            print_summary(fold_data, aggregated, fp=f)
            interpret_results(fold_data, aggregated, fp=f)
    else:
        print_summary(fold_data, aggregated)
        interpret_results(fold_data, aggregated)

    # Generate plots if requested
    if args.plot:
        plot_output = args.run_dir / "metrics_plots"
        plot_metrics(fold_data, aggregated, plot_output)

    return 0


if __name__ == "__main__":
    exit(main())

# python scripts/analyze_mlflow_runs.py /media/smbshare/srs-9/prl_project/training/roi_train2/run3 --distributed --plot --outfile $PRL_PROJECT_ROOT/scratch/roi_train2_run3_mlflow.txt -f 0 -f 1 -f 2