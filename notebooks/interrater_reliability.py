# %%
from pathlib import Path
import os
import pandas as pd
# from helpers.utils import dice_score
from helpers import paths
from helpers.paths import (
    RESOURCES_DIR, TRAIN_ROOT, PROJECT_ROOT, DATA_ROOT
)
from helpers.shell_interface import open_itksnap_workspace_cmd

import preprocessing.create_rois
from preprocessing.create_rois import ensure_ring_seg
from helpers.shell_interface import command, run_if_missing
from helpers.utils import dice_score


subject_sessions = pd.read_csv(RESOURCES_DIR/"subject-sessions.csv", index_col="sub")
subjects_file = PROJECT_ROOT / "training/roi_train2/subjects.txt"
with open(subjects_file, 'r') as f:
    subjects = [line.strip() for line in f.readlines()]
    
prl_df = pd.read_csv(
    "/home/srs-9/Projects/prl_project/src/resources/PRL_spreadsheet-lstai_update_label_reference.csv",
    index_col="subid"
)

from collections import defaultdict
dataroot = Path("/media/smbshare/3Tpioneer_bids")
label_names = [
    "lesion.t3m20/prl_mask_def_prob_LR.nii.gz", 
    "lesion.t3m20/prl_mask_def_prob_CH.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_SRS.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_SRS_CH.nii.gz",
]

labels_to_check = defaultdict(list)
for subid in subjects:
    sesid = subject_sessions.loc[int(subid), "ses"]
    subject_root = dataroot / f"sub-ms{subid}" / f"ses-{sesid}"
    for n in label_names:
        p = subject_root / n
        if p.exists():
            labels_to_check[subid].append(p)
            
for k,v in labels_to_check.copy().items():
    if len(v) < 2:
        labels_to_check.pop(k)

# %%



work_home = Path("/home/srs-9/Projects/prl_project/notebooks/interrater_reliability")
work_home.mkdir(exist_ok=True)
expand_xy, expand_z = 20, 2
expand_suffix = f"xy{expand_xy}_z{expand_z}"

subid = 2026
prl_initials = ["CH", "SRS"]
ses = str(subject_sessions.loc[subid, 'ses'])
subject_root = DATA_ROOT / f"sub{subid}-{ses}"
work_folder = work_home / subject_root.name
work_folder.mkdir(exist_ok=True)

bounding_box_file = (
        subject_root / f"lstai_bounding_boxes_xy{expand_xy}_z{expand_z}.txt"
    )

with open(bounding_box_file, "r") as f:
    bounding_boxes = []
    for line in f.readlines():
        parts = line.split(" ")
        bounding_boxes.append((int(parts[0]), " ".join(parts[1:]).strip()))

# PRL1
lesion_ind = int(prl_df.loc[subid, "PRL1_label"])
print(lesion_ind)
box = bounding_boxes[lesion_ind-1][1]
lesion_folder = work_folder / str(lesion_ind)
lesion_folder.mkdir(exist_ok=True)
labels_to_compare = []
for suffix in prl_initials:
    lesion_in = ensure_ring_seg(subject_root, suffix=suffix, output_dir=work_folder)
    lesion_tmp = lesion_folder / (lesion_in.name.removesuffix(".nii.gz") + f"_{expand_suffix}_tmp.nii.gz")
    run_if_missing(lesion_tmp, f"fslroi {lesion_in} {lesion_tmp} {box}")
    lesion_final = lesion_folder / (lesion_in.name.removesuffix(".nii.gz") + f"_{expand_suffix}.nii.gz")
    run_if_missing(
        lesion_final,
        f"c3d {lesion_tmp} -binarize -scale 2 -o {lesion_final}",
    )
    lesion_tmp.unlink(missing_ok=True)
    labels_to_compare.append(lesion_final)

print(dice_score(labels_to_compare[0], labels_to_compare[1], 2,2))