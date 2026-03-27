#!/usr/bin/env python3
"""
Unified CLI for MLflow experiment management and analysis.

Commands:
  collate   Collate per-fold mlruns into a single top-level mlruns directory
  symlink   Build a combined mlruns directory by symlinking experiments from
            all runs listed in experiment_runs.txt
  analyze   Analyze metrics across folds for a single run directory

Usage:
  python mlflow_cli.py collate <run_dir>
  python mlflow_cli.py symlink <output_dir> [--runs-file <path>]
  python mlflow_cli.py analyze <run_dir> [--distributed] [--plot] [--outfile <path>] [-f FOLD ...]
"""

import argparse
import sys
from pathlib import Path

# Default runs registry path relative to this file
_DEFAULT_RUNS_FILE = Path(__file__).parent.parent / "resources" / "experiment_runs.txt"


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------

def cmd_collate(args):
    from collate_distributed_mlruns import collate_mlruns
    collate_mlruns(args.run_dir)


# ---------------------------------------------------------------------------
# symlink
# ---------------------------------------------------------------------------

def _load_runs_file(runs_file: Path) -> list[Path]:
    if not runs_file.exists():
        print(f"Error: runs file not found: {runs_file}", file=sys.stderr)
        sys.exit(1)
    paths = []
    for line in runs_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
    return paths


def _ensure_default_experiment(combined: Path):
    default = combined / "0"
    if not default.exists():
        default.mkdir()
        (default / "meta.yaml").write_text(
            "artifact_location: mlflow-artifacts:/0\nexperiment_id: '0'\n"
            "lifecycle_stage: active\nname: Default\n"
        )


def cmd_symlink(args):
    runs_file: Path = args.runs_file
    output_dir: Path = args.output_dir

    run_dirs = _load_runs_file(runs_file)
    if not run_dirs:
        print("No run directories found in runs file.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".trash").mkdir(exist_ok=True)
    _ensure_default_experiment(output_dir)

    linked = 0
    skipped = 0
    for run_dir in run_dirs:
        mlruns = run_dir / "mlruns"
        if not mlruns.exists():
            print(f"Warning: no mlruns/ found in {run_dir}, skipping")
            continue
        for exp_dir in sorted(mlruns.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name in ("0", ".trash"):
                continue
            link = output_dir / exp_dir.name
            if link.exists() or link.is_symlink():
                skipped += 1
                continue
            try:
                link.symlink_to(exp_dir.resolve())
                print(f"  Linked {exp_dir.name} ({run_dir.name}) -> {exp_dir}")
            except OSError as e:
                if e.errno == 95:  # Operation not supported (e.g. SMB share)
                    print(
                        f"\nError: symlinks not supported on this filesystem ({output_dir}).\n"
                        f"Use a local output directory instead, e.g.:\n"
                        f"  python mlflow_cli.py symlink ~/mlflow_experiments\n",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                raise
            linked += 1

    print(f"\nDone. {linked} experiment(s) linked, {skipped} already present.")
    print(f"Start UI with:\n  mlflow ui --backend-store-uri {output_dir}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args):
    # Import analysis functions from existing module
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_mlflow_runs import (
        analyze_distributed_mlruns,
        analyze_unified_mlruns,
        aggregate_metrics,
        print_summary,
        interpret_results,
        plot_metrics,
    )

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    which_folds = [int(f) for f in args.fold] if args.fold else None

    if args.distributed:
        print("Using distributed analysis...")
        fold_data = analyze_distributed_mlruns(run_dir, which_folds=which_folds)
    else:
        print("Using unified analysis...")
        fold_data = analyze_unified_mlruns(run_dir)

    if not fold_data:
        print("Error: no fold data found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fold_data)} fold(s)")
    aggregated = aggregate_metrics(fold_data)

    if args.outfile:
        out = args.outfile if args.outfile.is_absolute() else run_dir / args.outfile
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            print_summary(fold_data, aggregated, fp=f)
            interpret_results(fold_data, aggregated, fp=f)
        print(f"Saved to {out}")
    else:
        print_summary(fold_data, aggregated)
        interpret_results(fold_data, aggregated)

    if args.plot:
        plot_metrics(fold_data, aggregated, run_dir / "metrics_plots")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlflow_cli.py",
        description="MLflow experiment management and analysis tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- collate ---
    p_collate = sub.add_parser(
        "collate",
        help="Collate per-fold mlruns into a single top-level mlruns directory",
    )
    p_collate.add_argument("run_dir", type=Path, help="Path to the run directory (e.g. run2/)")
    p_collate.set_defaults(func=cmd_collate)

    # --- symlink ---
    p_symlink = sub.add_parser(
        "symlink",
        help="Build a combined mlruns directory from all runs listed in experiment_runs.txt",
    )
    p_symlink.add_argument("output_dir", type=Path, help="Directory to create symlinks in")
    p_symlink.add_argument(
        "--runs-file", type=Path, default=_DEFAULT_RUNS_FILE,
        help=f"Path to experiment runs list (default: {_DEFAULT_RUNS_FILE})",
    )
    p_symlink.set_defaults(func=cmd_symlink)

    # --- analyze ---
    p_analyze = sub.add_parser(
        "analyze",
        help="Analyze MLflow metrics across folds for a single run directory",
    )
    p_analyze.add_argument("run_dir", type=Path, help="Path to the run directory")
    p_analyze.add_argument(
        "--distributed", action="store_true",
        help="Read from per-fold mlruns (for run1/run2 before collation)",
    )
    p_analyze.add_argument("--plot", action="store_true", help="Generate metric plots")
    p_analyze.add_argument("--outfile", type=Path, default=None, help="Save output to file")
    p_analyze.add_argument(
        "-f", "--fold", action="append", default=None, metavar="N",
        help="Restrict to specific fold(s), e.g. -f 0 -f 1",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
