"""Create datalist_template.json with stratified fold assignments.

Scans subject directories to categorize PRL vs lesion-only cases, then
creates a stratified train/test split with fold assignments.
"""

import os
import random
import json
import argparse
import sys
import re
import pandas as pd
from pathlib import Path
from math import floor

from loguru import logger

from helpers.paths import load_config


def create_datalist_template(subjects, suffix_to_use, prl_df, data_root,
                              n_folds, test_split, output_path, rebuild=False):
    """Create datalist_template.json with stratified fold assignments.

    This is the primary API for Dataset.create_datalist(). All
    parameters are explicit — no globals. The template is image-agnostic:
    "image" stores the case directory (relative to data_root), not a
    stacked-image prefix. The image stack is determined later by
    Experiment.prepare_data().

    Args:
        subjects: List of subject IDs.
        suffix_to_use: Dict mapping subid → annotator suffix.
        prl_df: PRL spreadsheet DataFrame indexed by subid.
        data_root: Root data directory containing subject folders.
        n_folds: Number of cross-validation folds.
        test_split: Fraction of data for test set.
        output_path: Path to write datalist_template.json.
        rebuild: If False and output_path exists, skip.

    Returns:
        Path to the written datalist_template.json, or None if skipped.
    """
    output_path = Path(output_path)
    label_info_path = output_path.with_name("label_info.json")

    if output_path.exists() and not rebuild:
        logger.info(f"{output_path} exists; use rebuild=True to replace it")
        return None
    
    if suffix_to_use is None:
        suffix_to_use = {}
    for k, suffix in suffix_to_use:
        if len(suffix) > 1 and suffix[0] != "_":
            suffix_to_use[k] = "_" + suffix
    
    data_root = Path(data_root)

    prl_folders = []
    lesion_folders = []
    for subid in subjects:
        subid = int(subid)
        sesid = prl_df.loc[subid, "date_mri"]
        subject_root = data_root / f"sub{subid}-{sesid}"
        prl_labels = set([
            int(prl_df.loc[subid, f"PRL{i}_label"])
            for i in range(1, 21)
            if prl_df.loc[subid, f"confidence.{i-1}"] in ["definite", "probable"]
        ])

        folders = [
            Path(item.path) for item in os.scandir(subject_root)
            if item.is_dir() and re.match(r"^\d+", item.name)
        ]
        for folder in folders:
            index = int(folder.name)
            if index < 1:
                continue
            if index in prl_labels:
                prl_folders.append((folder, subid, index))
            else:
                lesion_folders.append((folder, subid, index))

    #FIXME can clean up by using rstrip("_") + "_"
    def _make_entry(folder, subid, index, case_type, suffix=""):
        if len(suffix) > 0 and suffix[0] != "_":
            suffix = "_" + suffix
        rel = str(folder.relative_to(data_root))
        if case_type == "PRL":
            label = f"{rel}/prl_label{suffix}_"
        else:
            label = f"{rel}/lesion_"
        return {
            "subid": subid, "lesion_index": index,
            "image": f"{rel}/",
            "label": label,
            "case_type": case_type,
        }

    datalist = {"training": [], "testing": []}

    # PRL cases
    inds = list(range(len(prl_folders)))
    random.shuffle(inds)
    test_end_ind = floor(len(inds) * test_split)
    for i in range(test_end_ind):
        ind = inds[i]
        folder, subid, index = prl_folders[ind]
        entry = _make_entry(folder, subid, index, "PRL", suffix_to_use.get(subid, ""))
        datalist["testing"].append(entry)
    for i in range(test_end_ind, len(inds)):
        fold = i % n_folds
        ind = inds[i]
        folder, subid, index = prl_folders[ind]
        entry = _make_entry(folder, subid, index, "PRL", suffix_to_use.get(subid, ""))
        entry["fold"] = fold
        datalist["training"].append(entry)

    # Lesion-only cases
    inds = list(range(len(lesion_folders)))
    random.shuffle(inds)
    test_end_ind = floor(len(inds) * test_split)
    for i in range(test_end_ind):
        ind = inds[i]
        folder, subid, index = lesion_folders[ind]
        entry = _make_entry(folder, subid, index, "Lesion")
        datalist["testing"].append(entry)
    for i in range(test_end_ind, len(inds)):
        fold = i % n_folds
        ind = inds[i]
        folder, subid, index = lesion_folders[ind]
        entry = _make_entry(folder, subid, index, "Lesion")
        entry["fold"] = fold
        datalist["training"].append(entry)

    with open(output_path, "w") as f:
        json.dump(datalist, f, indent=4)

    label_types = {
        "prl_labels": [
            str(item[0] / f"prl_label{suffix_to_use.get(item[1], '')}_")
            for item in prl_folders
        ],
        "lesion_labels": [str(item[0]) for item in lesion_folders],
    }
    with open(label_info_path, "w") as f:
        json.dump(label_types, f, indent=4)

    return output_path


# --- CLI wrapper (backward compatibility) ---

def make_argument_parser(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawTextHelpFormatter,
        description="Create datalist template with fold assignments",
    )
    parser.add_argument("label_config", type=str, help="Path to config file")
    parser.add_argument("monai_config", type=str, help="Path to config file")
    parser.add_argument(
        "--rebuild", action="store_true", default=False,
        help="To rebuild the template",
    )
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = make_argument_parser(argv)
    args, _ = parser.parse_known_args()

    monai_config = load_config(args.monai_config)
    label_config = load_config(args.label_config)

    train_home = Path(label_config["train_home"])
    data_root = Path(label_config["dataroot"])

    prl_df = pd.read_csv(label_config["prl_df"], index_col="subid")

    with open(label_config["subjects"], "r") as f:
        subjects = [int(line.strip()) for line in f.readlines()]

    suffix_to_use = {}
    with open(label_config["suffix_to_use"], "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            subid, suffix = line.strip().split(",")
            suffix_to_use[int(subid)] = suffix

    create_datalist_template(
        subjects=subjects,
        suffix_to_use=suffix_to_use,
        prl_df=prl_df,
        data_root=data_root,
        n_folds=monai_config["N_FOLDS"],
        test_split=monai_config["TEST_SPLIT"],
        output_path=train_home / "datalist_template.json",
        rebuild=args.rebuild,
    )


def main1():
    from core.dataset import Dataset
    inference_dataset = Dataset("inference_dataset")
    inference_dataset.create_datalist()

if __name__ == "__main__":
    main1()
