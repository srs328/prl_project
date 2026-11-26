import numpy as np

import subprocess
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

orig_index_df = pd.read_csv("/home/srs-9/Projects/prl_project/PRL_spreadsheet_label_reference.csv",
                         index_col="subid")
new_index_df = pd.read_csv("/home/srs-9/Projects/prl_project/PRL_spreadsheet-lstai_update_label_reference.csv",
                        index_col="subid")
dataroot = Path("/media/smbshare/3Tpioneer_bids")
work_dir = Path(__file__).parent / "tmp"
work_dir.mkdir(exist_ok=True)
mask = work_dir / "mask.nii.gz"
masked = work_dir / "masked.nii.gz"

prl_labels = [
    "lesion.t3m20/prl_mask_def_prob_LR.nii.gz" 
    "lesion.t3m20/prl_mask_def_prob_CH.nii.gz"
    "lesion.t3m20/prl_mask_def_prob_SRS.nii.gz"
]

orig_index_files = [
    "lesion.t3m20/mlesion_index.t3m20.nii.gz",
    "lesion.t3m20/centerlesion_analysis/mlesion_index.t3m20.nii.gz"
    "lesion.t3m20/lesion_index.t3m20.nii.gz",
    "lesion_index.t3m20.nii.gz"
]
new_index_file = "lst-ai/lstai_lesion_index.nii.gz"

for subid, row in tqdm(orig_index_df.iterrows(), total=len(orig_index_df)):
    sesid = row['date_mri']
    subject_root = dataroot / f"sub-ms{subid}/ses-{sesid}"
    for file in orig_index_files:
        orig_index_map = subject_root / file
        if orig_index_map.exists():
            break
    else:
        print(f"Could not find original index map for sub{subid}")
        continue
    print(f"Using {orig_index_map} as original lesion index")
    new_index_map = subject_root / new_index_file
    if not new_index_map.exists():
        lst_seg = subject_root / "lst-ai/space-flair_seg-lst.nii.gz"
        cmd = f"c3d {lst_seg} -comp -o {new_index_map}"
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error with c3d on {subid}")
            print(e.stderr)
            continue
    for i in range(1, 21):
        prl_index = row[f"PRL{i}_label"]
        if np.isnan(prl_index):
            continue
        
        try:
            cmd = f"fslmaths {orig_index_map} -uthr {prl_index} -thr {prl_index} -bin {work_dir}/mask.nii.gz"
            subprocess.run(cmd, shell=True, check=True)
            cmd = f"fslmaths {new_index_map} -mul {mask} {masked}"
            subprocess.run(cmd, shell=True, check=True)
            cmd = f"fslstats {masked} -R"
            result = subprocess.run(cmd, shell=True, capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed at {cmd} with:")
            print(e.stderr)
            continue
        ind = int(float(result.stdout.split(" ")[1]))
        print(f"sub{subid} PRL{i}_label is {ind}")
        new_index_df.loc[subid, f"PRL{i}_label"] = ind
        
new_index_df.to_csv("/home/srs-9/Projects/prl_project/PRL_spreadsheet-lstai_update_label_reference (double check).csv")