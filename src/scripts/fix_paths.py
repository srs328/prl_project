#!/usr/bin/env python3
"""Utilities for fixing paths and configs in existing run directories.

Commands:
  hyper_params  Replace a path substring in all hyper_parameters.yaml files
  label_configs Migrate label_config.json files to new train_home/dataset_name format

Usage:
    python fix_paths.py hyper_params /path/to/run_dir /home/shridhar.singh9-umw /media/smbshare/srs-9
    python fix_paths.py label_configs /path/to/training_root
"""

import argparse
import json
import sys
from pathlib import Path


def fix_paths(root: Path, old: str, new: str, dry_run: bool = False) -> int:
    """Replace old path prefix with new in all hyper_parameters.yaml files.

    Args:
        root: Run directory or parent experiment directory to search.
        old: Path substring to replace.
        new: Replacement string.
        dry_run: Print changes without writing.

    Returns:
        Number of files modified.
    """
    print("Inside fix_paths")
    count = 0
    for yaml_path in sorted(root.rglob("*.yaml")):
        print(yaml_path)
        text = yaml_path.read_text()
        if old not in text:
            continue

        updated = text.replace(old, new)
        if dry_run:
            print(f"[DRY RUN] {yaml_path}")
        else:
            yaml_path.write_text(updated)
            print(f"Updated {yaml_path}")
        count += 1

    return count


def migrate_label_configs(root: Path, dry_run: bool = False) -> int:
    """Migrate label_config.json files to new train_home/dataset_name format.

    Old format: "train_home": ".../training/roi_train2"
    New format: "train_home": ".../training", "dataset_name": "roi_train2"

    Args:
        root: Root directory to search recursively for label_config.json files.
        dry_run: Print changes without writing.

    Returns:
        Number of files modified.
    """
    count = 0
    for cfg_path in sorted(root.rglob("label_config.json")):
        with open(cfg_path) as f:
            config = json.load(f)

        if "dataset_name" in config:
            continue  # already migrated

        old_train_home = config.get("train_home", "")
        if not old_train_home:
            print(f"Skipping {cfg_path} — no train_home field")
            continue

        dataset_name = Path(old_train_home).name
        new_train_home = str(Path(old_train_home).parent)

        if dry_run:
            print(f"[DRY RUN] {cfg_path}")
            print(f"  train_home: {old_train_home!r} -> {new_train_home!r}")
            print(f"  + dataset_name: {dataset_name!r}")
        else:
            config["train_home"] = new_train_home
            config["dataset_name"] = dataset_name
            with open(cfg_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Updated {cfg_path}")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Fix paths and configs in existing run directories"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hp = subparsers.add_parser("yamls",
                               help="Replace path substring in any .yaml files")
    hp.add_argument("root", type=Path, help="Run or experiment directory to search")
    hp.add_argument("old", type=str, help="Path substring to replace")
    hp.add_argument("new", type=str, help="Replacement string")
    hp.add_argument("--dry-run", action="store_true")

    lc = subparsers.add_parser("label_configs",
                                help="Migrate label_config.json to new train_home/dataset_name format")
    lc.add_argument("root", type=Path, help="Root directory to search")
    lc.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not args.root.exists():
        print(f"Error: {args.root} not found")
        return 1

    if args.command == "yamls":
        count = fix_paths(args.root, args.old, args.new, dry_run=args.dry_run)
    else:
        count = migrate_label_configs(args.root, dry_run=args.dry_run)

    print(f"\n{'Would update' if args.dry_run else 'Updated'} {count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
