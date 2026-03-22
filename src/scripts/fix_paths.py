#!/usr/bin/env python3
"""Fix paths in hyper_parameters.yaml files after copying runs between machines.

Usage:
    python fix_paths.py /path/to/run_dir /home/shridhar.singh9-umw /media/smbshare/srs-9
    python fix_paths.py /path/to/experiment_dir /home/shridhar.singh9-umw /media/smbshare/srs-9
"""

import argparse
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
    for yaml_path in sorted(root.rglob("hyper_parameters.yaml")):
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


def main():
    parser = argparse.ArgumentParser(
        description="Fix paths in hyper_parameters.yaml after copying runs"
    )
    parser.add_argument("root", type=Path, help="Run or experiment directory to search")
    parser.add_argument("old", type=str, help="Path substring to replace")
    parser.add_argument("new", type=str, help="Replacement string")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    print("Before parse args")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Error: {args.root} not found")
        return 1
    print("Before fix_paths")
    count = fix_paths(args.root, args.old, args.new, dry_run=args.dry_run)
    print(f"\n{'Would update' if args.dry_run else 'Updated'} {count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
