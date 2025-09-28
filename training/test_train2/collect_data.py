#%%
from pathlib import Path
import os
import csv
import shutil

#%%

dataroot = Path("/media/smbshare/3Tpioneer_bids")
targetroot = Path("/media/smbshare/srs-9/prl_project/training/test_train2")

subject_sessions = []
with open("/home/srs-9/Projects/prl_project/data/subject-sessions.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    subject_sessions = [line for line in reader]


subjects = []
for sub_ses in subject_sessions:
    subject_root = dataroot / f"sub-ms{sub_ses[0]}/ses-{sub_ses[1]}"
    lesion_path = subject_root / "lesion.t3m20"
    src_label = lesion_path / "prl_mask_def_prob_LR.nii.gz"
    if not src_label.exists():
        continue

    dst_label = targetroot / "labelsTr" / f"sub{sub_ses[0]}.nii.gz"
    if not dst_label.exists():
        shutil.copyfile(src_label, dst_label)
    
    src_img = subject_root / "flair.phase.t1.nii.gz"
    dst_img = targetroot / "imagesTr" / f"sub{sub_ses[0]}.nii.gz"
    if not dst_img.exists():
        shutil.copyfile(src_img, dst_img)

    subjects.append(sub_ses[0])

with open("subjects.txt", 'w') as f:
    for sub in subjects:
        f.write(sub + "\n")
