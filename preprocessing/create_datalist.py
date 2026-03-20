import os
import random
import json
import argparse
import sys
import pandas as pd
from pathlib import Path
import re
from math import floor
from helpers.paths import load_config
    
    
def make_argument_parser(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawTextHelpFormatter,
        description="Create ROIs",
    )
    parser.add_argument("label_config", type=str, help="Path to config file")
    parser.add_argument("monai_config", type=str, help="Path to config file")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="To rebuild the template",
    )

    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = make_argument_parser(argv)
    args, _ = parser.parse_known_args()

    monai_config = load_config(args.monai_config)
    N_FOLDS = monai_config['N_FOLDS']
    TEST_SPLIT = monai_config['TEST_SPLIT']

    label_config = load_config(args.label_config)
        
    train_home = Path(label_config['train_home'])
    datalist_template_path = train_home/"datalist_template.json"
    label_info_path = train_home/"label_info.json"
    
    if datalist_template_path.exists() and not args.rebuild:
        print(f"{datalist_template_path} exists; use --rebuild replace it")
        return
    
    suffix_to_use = {}
    with open(label_config["suffix_to_use"], "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            subid, suffix = line.strip().split(",")
            suffix_to_use[int(subid)] = suffix

    global dataroot
    dataroot = Path(label_config["dataroot"])
    
    image_basenames = [im.removesuffix(".nii.gz") for im in sorted(label_config['images'])]
    stacked_image_prefix = ".".join(image_basenames) + "_"
    
    prl_df = pd.read_csv(label_config["prl_df"], index_col="subid")

    with open(label_config["subjects"], "r") as f:
        subjects = [int(line.strip()) for line in f.readlines()]

    prl_folders = []
    lesion_folders = []
    datalist = {"training": [], "testing": []}
    for subid in subjects:
        subid = int(subid)
        sesid = prl_df.loc[subid, "date_mri"]
        subject_root = dataroot / f"sub{subid}-{sesid}"
        prl_labels = set([
            int(prl_df.loc[subid, f"PRL{i}_label"])
            for i in range(1, 21)
            if prl_df.loc[subid, f"confidence.{i-1}"] in ["definite", "probable"]
        ])
        
        folders = [Path(item.path) for item in os.scandir(subject_root) if item.is_dir() and re.match(r"^\d+", item.name)]
        for folder in folders:
            index = int(folder.name)
            if index < 1:
                continue
            if index in prl_labels:
                prl_folders.append((folder, subid, index))
            else:
                lesion_folders.append((folder, subid, index))
    
    inds = list(range(len(prl_folders)))
    random.shuffle(inds)
    test_end_ind = floor(len(inds) * TEST_SPLIT)
    for i in range(test_end_ind):
        ind = inds[i]
        folder, subid, index = prl_folders[ind]
        suffix = suffix_to_use[subid]
        datalist['testing'].append(
            {"subid": subid, "lesion_index": index, "image": str(folder/stacked_image_prefix), "label": str(folder/f"prl_label_{suffix}_")}
        )
    for i in range(test_end_ind, len(inds)):
        fold = i%N_FOLDS
        ind = inds[i]
        folder, subid, index = prl_folders[ind]
        suffix = suffix_to_use[subid]
        datalist['training'].append(
            {
                "subid": subid, "lesion_index": index,
                "fold": fold,
                "image": str(folder/stacked_image_prefix), 
                "label": str(folder/f"prl_label_{suffix}_")
            }
        )
    
    inds = list(range(len(lesion_folders)))
    random.shuffle(inds)
    test_end_ind = floor(len(inds) * TEST_SPLIT)
    for i in range(test_end_ind):
        ind = inds[i]
        folder, subid, index = lesion_folders[ind]
        datalist['testing'].append(
            {"subid": subid, "lesion_index": index, "image": str(folder/stacked_image_prefix), "label": str(folder/"lesion_")}
        )
    for i in range(test_end_ind, len(inds)):
        fold = i%N_FOLDS
        ind = inds[i]
        folder, subid, index = lesion_folders[ind]
        datalist['training'].append(
            {
                "subid": subid, "lesion_index": index,
                "fold": fold,
                "image": str(folder/stacked_image_prefix), 
                "label": str(folder/"lesion_")
            }
        )
        
    with open(datalist_template_path, 'w') as f:
        json.dump(datalist, f, indent=4)

    label_types = {
        "prl_labels": [str(item[0]/f"prl_label_{suffix_to_use[item[1]]}_") for item in prl_folders],
        "lesion_labels": [str(item[0]) for item in lesion_folders]
    }

    with open(label_info_path, 'w') as f:
        json.dump(label_types, f, indent=4)
        
        
if __name__ == "__main__":
    main()