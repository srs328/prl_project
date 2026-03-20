"""
Example usage: python scripts/check_prl_case.py /media/smbshare/srs-9/prl_project/training/roi_train2/run2
"""

from collections import defaultdict
import sys
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from functools import partial
import json

sys.path.append("/home/srs-9/Projects/prl_project")
from helpers.shell_interface import command, run_if_missing, open_itksnap_workspace_cmd, open_fsleyes_cmd
from helpers.parallel import BetterPool
from helpers.paths import load_config
from preprocessing.verify_segmentations import verify_prl, verify_lesion


def main():
    parser = argparse.ArgumentParser(description="Open a PRL case")
    parser.add_argument("train_dir", type=Path, help="Path to run directory")
    parser.add_argument("--fsleyes", action="store_true", default=False, help="To use fsleyes")
    parser.add_argument("--output", type=Path, help="Where to save the commands list", default=None)
    args = parser.parse_args()
    
    if not args.train_dir.exists():
        print(f"Error: Run directory not found: {args.train_dir}")
        return 1
    
    train_dir = Path(args.train_dir)
    with open(train_dir/"label_config.json", 'r') as f:
        label_config = json.load(f)
    
    expand_xy, expand_z = label_config['expand_xy'], label_config['expand_z']
    with open(train_dir/f"datalist_xy{expand_xy}_z{expand_z}.json", 'r') as f:
        datalist = json.load(f)
        
    prl_cases = defaultdict(list)
    
    # print(len(datalist['training']))
    for case in datalist['training']:
        label = Path(case['label'])
        if "prl" not in label.name:
            continue
        # print(label.name)
        folder = label.parent
        images = [folder / f"{im}_xy{expand_xy}_z{expand_z}.nii.gz"
                  for im in label_config['images']]
        labels = [label]
        
        if args.fsleyes:
            cmd = open_fsleyes_cmd(images, labels=labels, rename_root=("/media/smbshare", "/mnt/z"))
        else:
            cmd = open_itksnap_workspace_cmd(images, labels=labels, rename_root=("/media/smbshare", "Z:/"))
        prl_cases[case['fold']].append(cmd)
    
    output = args.output
    if output is None:
        output = train_dir / "open_prl_cases_cmds.txt"
    print(output)
    with open(output, 'w') as f:
        for fold in prl_cases:
            f.write(f"FOLD {fold}\n")
            f.write("---"*20 + "\n")
            for cmd in prl_cases[fold]:
                f.write(cmd + "\n")
    
if __name__ == "__main__":
    main()