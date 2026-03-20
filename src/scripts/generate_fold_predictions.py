#!/usr/bin/env python3
"""
Generate validation predictions for each fold using the Auto3DSeg infrastructure.

This script leverages the auto-generated inference code from MONAI Auto3DSeg
to run inference on validation data for each fold and organize results.

Usage:
    python generate_fold_predictions.py /path/to/run2
    python generate_fold_predictions.py /path/to/run2 --fold 4
    python generate_fold_predictions.py /path/to/run2 --output-dir /custom/predictions
"""

import json
import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import shutil

from helpers.paths import load_config


def get_validation_cases(datalist_path: Path, fold_num: int) -> List[Dict]:
    """Get all validation cases for a specific fold from the datalist."""
    with open(datalist_path, 'r') as f:
        datalist = json.load(f)

    return [item for item in datalist.get('training', []) if item.get('fold') == fold_num]


def create_fold_datalist(validation_cases: List[Dict], output_path: Path) -> None:
    """Create a temporary datalist with only validation cases for inference."""
    datalist = {
        "testing": validation_cases,  # Treat validation cases as "testing" for inference
    }

    with open(output_path, 'w') as f:
        json.dump(datalist, f, indent=2)


def run_fold_inference(run_dir: Path, fold_num: int, datalist_path: Path,
                      data_root: Path, output_dir: Path) -> bool:
    """
    Run inference for a single fold using Auto3DSeg infrastructure.

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running inference for Fold {fold_num}")
    print(f"{'='*80}\n")

    # Get validation cases
    validation_cases = get_validation_cases(datalist_path, fold_num)

    if not validation_cases:
        print(f"No validation cases found for fold {fold_num}")
        return False

    print(f"Found {len(validation_cases)} validation cases")

    # Create fold-specific output directory
    fold_output_dir = output_dir / f"fold{fold_num}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary datalist with just validation cases
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_datalist = Path(f.name)
        create_fold_datalist(validation_cases, temp_datalist)

    try:
        # Path to the fold's auto-generated scripts
        fold_scripts_dir = run_dir / f"segresnet_{fold_num}" / "scripts"
        segmenter_path = fold_scripts_dir / "segmenter.py"

        if not segmenter_path.exists():
            print(f"Error: Segmenter script not found at {segmenter_path}")
            return False

        # Create config overrides for inference
        config_file = run_dir / f"segresnet_{fold_num}" / "configs" / "hyper_parameters.yaml"

        if not config_file.exists():
            print(f"Error: Config file not found at {config_file}")
            return False

        # Run inference using the auto-generated segmenter
        # This will use the infer.py infrastructure
        infer_script = fold_scripts_dir / "infer.py"

        if not infer_script.exists():
            print(f"Error: Infer script not found at {infer_script}")
            return False

        print(f"\nRunning inference from: {infer_script}")
        print(f"Config: {config_file}")
        print(f"Output dir: {fold_output_dir}")

        # Build command to run inference
        # The infer.py script uses fire, so we can pass parameters directly
        cmd = [
            sys.executable,
            str(infer_script),
            "run",
            f"--config_file={config_file}",
            f"--infer#output_path={fold_output_dir}",
            f"--data_list_file_path={temp_datalist}",
        ]

        print(f"\nCommand: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, cwd=str(fold_scripts_dir))

        if result.returncode != 0:
            print(f"Error: Inference failed with return code {result.returncode}")
            return False

        print(f"\n✓ Fold {fold_num} inference complete")
        print(f"  Results saved to: {fold_output_dir}")

        return True

    finally:
        # Clean up temporary datalist
        if temp_datalist.exists():
            temp_datalist.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Generate validation predictions for each fold"
    )
    parser.add_argument("train_dir", type=Path, help="Path to run directory (e.g., run2)")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold to process (default: all)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: train_dir/fold_predictions)")

    args = parser.parse_args()

    if not args.train_dir.exists():
        print(f"Error: Run directory not found: {args.train_dir}")
        return 1

    run_dir = args.train_dir
    label_config = load_config(run_dir / "label_config.json")
    expand_xy, expand_z = label_config["expand_xy"], label_config["expand_z"]
    datalist_path = run_dir / f"datalist_xy{expand_xy}_z{expand_z}.json"
    data_root = Path(label_config["dataroot"])

    if not datalist_path.exists():
        print(f"Error: Datalist not found: {datalist_path}")
        return 1

    # Set default output directory
    output_dir = args.output_dir or (run_dir / "fold_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(datalist_path, 'r') as f:
        datalist = json.load(f)

    print(f"Run directory: {run_dir}")
    print(f"Datalist: {datalist_path}")
    print(f"Output directory: {output_dir}")

    # Determine which folds to process
    if args.fold is not None:
        folds = [args.fold]
    else:
        fold_nums = set(item.get('fold') for item in datalist.get('training', []))
        folds = sorted(fold_nums)

    print(f"Folds to process: {folds}")

    # Run inference for each fold
    results = {}
    for fold_num in folds:
        try:
            success = run_fold_inference(run_dir, fold_num, datalist_path, data_root, output_dir)
            results[fold_num] = "success" if success else "failed"
        except Exception as e:
            print(f"Error processing fold {fold_num}: {e}")
            results[fold_num] = f"error: {e}"

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for fold_num, result in sorted(results.items()):
        status = "✓" if result == "success" else "✗"
        print(f"{status} Fold {fold_num}: {result}")

    print(f"\nPredictions saved to: {output_dir}/")

    return 0 if all(r == "success" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
