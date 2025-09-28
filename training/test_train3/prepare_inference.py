#%%
from pathlib import Path
import os
import csv
import shutil
import subprocess

#%%

dataroot = Path("/media/smbshare/3Tpioneer_bids")
targetroot = Path("/media/smbshare/srs-9/prl_project/training/test_train3")

with open("/home/srs-9/Projects/prl_project/data/subject-sessions.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    subject_sessions = {line[0]: line[1] for line in reader}

#%%
subjects = [2095, 2087, 2086, 2060, 2041]
subjects2 = [1396, 1038, 2041]
filepaths = []
for subid in subjects2:
    sub = str(subid)
    ses = subject_sessions[sub]
    subject_root = dataroot / f"sub-ms{sub}/ses-{ses}"
    src = subject_root / "flair.phase.t1.nii.gz"
    dst = targetroot / "imagesTs" / f"sub{sub}.nii.gz"
    shutil.copyfile(src, dst)
    filepaths.append(str(dst))

    lesion_path = subject_root / "lesion.t3m20"
    src_label = lesion_path / "prl_mask_def_prob_CH.nii.gz"
    dst_label = targetroot / "labelsGt" / f"sub{sub}.nii.gz"
    lst_seg = subject_root / "lst-ai/space-flair_seg-lst.nii.gz"
    cmd = ["fslmaths", str(src_label), "-binv", "-mul", str(lst_seg), "-add", str(src_label), str(dst_label)]
    subprocess.run(cmd)


with open("inference_files2.txt", 'w') as f:
    for path in filepaths:
        f.write(path + "\n")

#%%

subjects = []
for sub_ses in subject_sessions:
    subject_root = dataroot / f"sub-ms{sub_ses}/ses-{subject_sessions[sub_ses]}"
    lesion_path = subject_root / "lesion.t3m20"
    src_label = lesion_path / "prl_mask_def_prob_CH.nii.gz"
    if src_label.exists():
        print(sub_ses)

#     if not src_label.exists():
#         continue

#     dst_label = targetroot / "labelsGt" / f"sub{sub_ses[0]}.nii.gz"
#     lst_seg = subject_root / "lst-ai/space-flair_seg-lst.nii.gz"
#     cmd = ["fslmaths", str(src_label), "-binv", "-mul", str(lst_seg), "-add", str(src_label), str(dst_label)]
#     subprocess.run(cmd)
#     assert dst_label.exists()

#     src_img = subject_root / "flair.phase.t1.nii.gz"
#     dst_img = targetroot / "imagesTs" / f"sub{sub_ses[0]}.nii.gz"
#     if not dst_img.exists():
#         shutil.copyfile(src_img, dst_img)

#     subjects.append(str(dst_img))

# with open("inference_files2.txt", 'w') as f:
#     for sub in subjects:
#         f.write(sub + "\n")
