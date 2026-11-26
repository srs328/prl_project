#!/usr/bin/env python3
"""
Diagnose fold differences using MONAI datastats and performance metrics.

Loads datastats_by_case.yaml (per-case image/label statistics from Auto3DSeg),
reconciles with the datalist to organize by fold/split, and optionally joins
with performance_metrics.csv to correlate data characteristics with model output.

Usage:
    python scripts/diagnose_fold_differences.py /path/to/run2
    python scripts/diagnose_fold_differences.py /path/to/run2 --problem-fold 4
    python scripts/diagnose_fold_differences.py /path/to/run2 --output-csv fold_analysis.csv
"""

import sys
import re
import json
import argparse
from pathlib import Path
from math import prod

import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.paths import load_config


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_case_stats(case: dict) -> dict:
    """Extract flat features from a single datastats_by_case entry."""
    img = case["image_stats"]
    sizemm = img["sizemm"]
    volume_mm3 = prod(sizemm)

    # Shape (take first channel, they're identical)
    shape = img["shape"][0] if isinstance(img["shape"][0], list) else img["shape"]

    # Per-channel intensity
    intensities = img["intensity"]
    ch0 = intensities[0] if isinstance(intensities, list) else intensities
    ch1 = intensities[1] if isinstance(intensities, list) and len(intensities) > 1 else {}

    # Label stats
    label_stats = case.get("label_stats", {})
    labels_present = label_stats.get("labels", [])
    label_entries = label_stats.get("label", [])

    lesion_fg_pct = np.nan
    rim_fg_pct = np.nan
    if len(label_entries) >= 2:
        lesion_fg_pct = label_entries[1].get("foreground_percentage", np.nan)
    if len(label_entries) >= 3:
        rim_fg_pct = label_entries[2].get("foreground_percentage", np.nan)

    return {
        "image_filepath": case["image_filepath"],
        "label_filepath": case["label_filepath"],
        "volume_mm3": volume_mm3,
        "shape_x": shape[0],
        "shape_y": shape[1],
        "shape_z": shape[2],
        "sizemm_x": sizemm[0],
        "sizemm_y": sizemm[1],
        "sizemm_z": sizemm[2],
        "flair_mean": ch0.get("mean", np.nan),
        "flair_std": ch0.get("stdev", np.nan),
        "flair_median": ch0.get("median", np.nan),
        "flair_min": ch0.get("min", np.nan),
        "flair_max": ch0.get("max", np.nan),
        "phase_mean": ch1.get("mean", np.nan),
        "phase_std": ch1.get("stdev", np.nan),
        "phase_median": ch1.get("median", np.nan),
        "phase_min": ch1.get("min", np.nan),
        "phase_max": ch1.get("max", np.nan),
        "n_labels": len(labels_present),
        "lesion_fg_pct": lesion_fg_pct,
        "rim_fg_pct": rim_fg_pct,
    }


def extract_subid_index(filepath: str):
    """Extract subid and lesion_index from a datastats image filepath."""
    m = re.search(r"sub(\d+)-\d+/(\d+)/", filepath)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# ---------------------------------------------------------------------------
# Loading & joining
# ---------------------------------------------------------------------------

def load_datastats(run_dir: Path) -> pd.DataFrame:
    """Parse datastats_by_case.yaml into a DataFrame of per-case features."""
    yaml_path = run_dir / "datastats_by_case.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found")
        sys.exit(1)

    print(f"Loading {yaml_path} ...")
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cases = raw["stats_by_cases"]
    rows = [parse_case_stats(c) for c in cases]
    df = pd.DataFrame(rows)

    # Extract subid and lesion_index from filepath
    ids = df["image_filepath"].apply(lambda p: pd.Series(extract_subid_index(p), index=["subid", "lesion_index"]))
    df = pd.concat([ids, df], axis=1)

    print(f"  Loaded {len(df)} cases from datastats")
    return df


def load_datalist(run_dir: Path, label_config: dict) -> pd.DataFrame:
    """Load the datalist and return a DataFrame with fold/split assignments."""
    expand_xy = label_config["expand_xy"]
    expand_z = label_config["expand_z"]
    datalist_path = run_dir / f"datalist_xy{expand_xy}_z{expand_z}.json"

    with open(datalist_path) as f:
        datalist = json.load(f)

    rows = []
    for item in datalist.get("testing", []):
        rows.append({
            "image_filepath": item["image"],
            "subid": item.get("subid"),
            "lesion_index": item.get("lesion_index"),
            "split": "testing",
            "fold": np.nan,
            "case_type": "PRL" if "prl" in Path(item["label"]).name else "Lesion",
        })
    for item in datalist.get("training", []):
        rows.append({
            "image_filepath": item["image"],
            "subid": item.get("subid"),
            "lesion_index": item.get("lesion_index"),
            "split": "training",
            "fold": item.get("fold"),
            "case_type": "PRL" if "prl" in Path(item["label"]).name else "Lesion",
        })

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} entries from datalist ({df['split'].value_counts().to_dict()})")
    return df


def load_performance_metrics(run_dir: Path) -> pd.DataFrame | None:
    """Load performance_metrics.csv if it exists."""
    csv_path = run_dir / "performance_metrics.csv"
    if not csv_path.exists():
        print(f"  No performance_metrics.csv found (skipping metric join)")
        return None

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from performance_metrics.csv")
    return df


def build_joined_df(run_dir: Path, label_config: dict) -> pd.DataFrame:
    """Join datastats, datalist, and performance metrics into one DataFrame."""
    stats_df = load_datastats(run_dir)
    datalist_df = load_datalist(run_dir, label_config)

    # Join on image_filepath — outer so testing cases (not in datastats) are included
    # Drop subid/lesion_index from stats_df since datalist_df has them from the datalist directly
    stats_cols_to_drop = [c for c in ["subid", "lesion_index"] if c in stats_df.columns]
    df = datalist_df.merge(stats_df.drop(columns=stats_cols_to_drop, errors="ignore"), on="image_filepath", how="left")

    n_with_stats = df["volume_mm3"].notna().sum()
    n_without = df["volume_mm3"].isna().sum()
    print(f"  Joined: {n_with_stats} cases with datastats, {n_without} without (typically test cases)")

    # Join performance metrics if available
    perf_df = load_performance_metrics(run_dir)
    if perf_df is not None:
        perf_cols = ["subid", "lesion_index", "sensitivity", "specificity",
                     "precision", "f1", "fpr", "fnr", "accuracy",
                     "tp", "fp", "tn", "fn"]
        perf_subset = perf_df[[c for c in perf_cols if c in perf_df.columns]].copy()
        df = df.merge(perf_subset, on=["subid", "lesion_index"], how="left", suffixes=("", "_perf"))

    # Create a group label for display
    df["group"] = df.apply(
        lambda r: "testing" if r["split"] == "testing" else f"fold{int(r['fold'])}" if pd.notna(r["fold"]) else "unknown",
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize_group(group_df: pd.DataFrame, label: str) -> dict:
    """Compute summary statistics for a group of cases."""
    n = len(group_df)
    n_prl = (group_df["case_type"] == "PRL").sum()
    n_lesion = (group_df["case_type"] == "Lesion").sum()

    summary = {
        "group": label,
        "n_total": n,
        "n_prl": n_prl,
        "n_lesion": n_lesion,
    }

    for col in ["volume_mm3", "flair_mean", "flair_std", "phase_mean", "phase_std",
                "lesion_fg_pct", "rim_fg_pct", "shape_x", "shape_y", "shape_z"]:
        if col in group_df.columns:
            vals = group_df[col].dropna()
            if len(vals) > 0:
                summary[f"{col}_mean"] = vals.mean()
                summary[f"{col}_std"] = vals.std()
                summary[f"{col}_median"] = vals.median()
                summary[f"{col}_min"] = vals.min()
                summary[f"{col}_max"] = vals.max()

    # Performance metrics (PRL only — sensitivity undefined for lesion-only)
    prl_df = group_df[group_df["case_type"] == "PRL"]
    for metric in ["sensitivity", "specificity", "precision", "f1", "accuracy"]:
        if metric in prl_df.columns:
            vals = prl_df[metric].dropna()
            if len(vals) > 0:
                summary[f"prl_{metric}_mean"] = vals.mean()
                summary[f"prl_{metric}_std"] = vals.std()

    return summary


def print_report(df: pd.DataFrame, problem_fold: int | None = None):
    """Print the fold analysis report."""
    groups = ["testing"] + [f"fold{i}" for i in range(5)]
    existing_groups = [g for g in groups if g in df["group"].unique()]

    print("\n" + "=" * 100)
    print("FOLD ANALYSIS REPORT")
    print("=" * 100)

    # --- Distribution table ---
    print("\nDATA DISTRIBUTION")
    print("-" * 100)
    header = f"{'Group':<12} {'Total':>6} {'PRL':>5} {'Lesion':>7}   {'Vol(mm3) mean +/- std':<26} {'Rim fg% mean +/- std':<24}"
    print(header)
    print("-" * 100)

    summaries = []
    def fmt_mean_std(s, key, decimals=1):
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in s and std_key in s:
            return f"{s[mean_key]:.{decimals}f} +/- {s[std_key]:.{decimals}f}"
        return "N/A"

    for g in existing_groups:
        gdf = df[df["group"] == g]
        s = summarize_group(gdf, g)
        summaries.append(s)

        vol_str = fmt_mean_std(s, "volume_mm3", 1)
        rim_str = fmt_mean_std(s, "rim_fg_pct", 5)

        # Flag outliers
        flag = ""
        if problem_fold is not None and g == f"fold{problem_fold}":
            flag = "  <-- PROBLEM FOLD"

        print(f"{g:<12} {s['n_total']:>6} {s['n_prl']:>5} {s['n_lesion']:>7}   {vol_str:<26} {rim_str:<24}{flag}")

    # --- Intensity table ---
    print("\n\nINTENSITY BY GROUP")
    print("-" * 100)
    print(f"{'Group':<12} {'FLAIR mean +/- std':<28} {'Phase mean +/- std':<28}")
    print("-" * 100)
    for s in summaries:
        flair_str = fmt_mean_std(s, "flair_mean", 1)
        phase_str = fmt_mean_std(s, "phase_mean", 1)
        print(f"{s['group']:<12} {flair_str:<28} {phase_str:<28}")

    # --- Shape table ---
    print("\n\nSHAPE BY GROUP (voxels)")
    print("-" * 100)
    print(f"{'Group':<12} {'X mean +/- std':<22} {'Y mean +/- std':<22} {'Z mean +/- std':<22}")
    print("-" * 100)
    for s in summaries:
        x_str = fmt_mean_std(s, "shape_x", 1)
        y_str = fmt_mean_std(s, "shape_y", 1)
        z_str = fmt_mean_std(s, "shape_z", 1)
        print(f"{s['group']:<12} {x_str:<22} {y_str:<22} {z_str:<22}")

    # --- Performance table (PRL only) ---
    has_perf = any(s.get("prl_sensitivity_mean") is not None for s in summaries)
    if has_perf:
        print("\n\nPERFORMANCE BY GROUP (PRL cases only)")
        print("-" * 100)
        print(f"{'Group':<12} {'N PRL':>6} {'Sens mean':>10} {'Spec mean':>10} {'Prec mean':>10} {'F1 mean':>10}")
        print("-" * 100)
        for s in summaries:
            if s["n_prl"] > 0 and s.get("prl_sensitivity_mean") is not None:
                flag = ""
                if problem_fold is not None and s["group"] == f"fold{problem_fold}":
                    flag = "  <-- PROBLEM"
                print(
                    f"{s['group']:<12} {s['n_prl']:>6} "
                    f"{s.get('prl_sensitivity_mean', 0):>10.4f} "
                    f"{s.get('prl_specificity_mean', 0):>10.4f} "
                    f"{s.get('prl_precision_mean', 0):>10.4f} "
                    f"{s.get('prl_f1_mean', 0):>10.4f}{flag}"
                )

    # --- Outlier cases ---
    print("\n\nOUTLIER CASES (volume > 2 sigma above mean)")
    print("-" * 100)
    vol_mean = df["volume_mm3"].mean()
    vol_std = df["volume_mm3"].std()
    threshold = vol_mean + 2 * vol_std
    outliers = df[df["volume_mm3"] > threshold].sort_values("volume_mm3", ascending=False)

    if len(outliers) == 0:
        print("  No outlier cases found")
    else:
        for _, row in outliers.head(20).iterrows():
            f1_str = f"F1={row['f1']:.3f}" if "f1" in row and pd.notna(row.get("f1")) else ""
            print(
                f"  Sub{int(row['subid'])}-L{int(row['lesion_index'])} "
                f"({row['group']}): {row['volume_mm3']:.0f} mm3, "
                f"{row['case_type']} {f1_str}"
            )

    # --- Problem fold deep dive ---
    if problem_fold is not None:
        fold_label = f"fold{problem_fold}"
        if fold_label in df["group"].values:
            print(f"\n\nDEEP DIVE: FOLD {problem_fold}")
            print("-" * 100)
            fold_df = df[df["group"] == fold_label]
            others_df = df[(df["split"] == "training") & (df["group"] != fold_label)]

            # Compare distributions
            for feature in ["volume_mm3", "flair_mean", "phase_mean", "lesion_fg_pct", "rim_fg_pct"]:
                fold_vals = fold_df[feature].dropna()
                other_vals = others_df[feature].dropna()
                if len(fold_vals) > 0 and len(other_vals) > 0:
                    diff_pct = (fold_vals.mean() - other_vals.mean()) / other_vals.mean() * 100
                    direction = "higher" if diff_pct > 0 else "lower"
                    print(
                        f"  {feature:<20}: fold{problem_fold} mean={fold_vals.mean():.4f}, "
                        f"others mean={other_vals.mean():.4f} "
                        f"({abs(diff_pct):.1f}% {direction})"
                    )

            # Subjects unique to problem fold
            fold_subs = set(fold_df["subid"].dropna().astype(int))
            other_subs = set(others_df["subid"].dropna().astype(int))
            unique_subs = fold_subs - other_subs
            if unique_subs:
                print(f"\n  Subjects ONLY in fold {problem_fold}: {sorted(unique_subs)}")
            print(f"  Subjects in fold {problem_fold}: {sorted(fold_subs)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnose fold differences using MONAI datastats")
    parser.add_argument("train_dir", type=Path, help="Path to run directory (e.g., run2)")
    parser.add_argument("--problem-fold", type=int, default=None, help="Fold to investigate in detail")
    parser.add_argument("--output-csv", type=Path, default=None, help="Save per-case joined data to CSV")
    args = parser.parse_args()

    if not args.train_dir.exists():
        print(f"ERROR: {args.train_dir} not found")
        return 1

    run_dir = args.train_dir
    label_config = load_config(run_dir / "label_config.json")

    df = build_joined_df(run_dir, label_config)
    print_report(df, problem_fold=args.problem_fold)

    if args.output_csv:
        output_path = args.output_csv
        if not output_path.is_absolute():
            output_path = run_dir / output_path
        df.to_csv(output_path, index=False)
        print(f"\nPer-case data saved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
