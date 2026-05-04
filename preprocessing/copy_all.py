from pathlib import Path
import shutil
import pandas as pd
from helpers.paths import load_config, PROJECT_ROOT, DATA_ROOT
from tqdm import tqdm
import subprocess

label_config = load_config(PROJECT_ROOT / "training/roi_train2/label_config.jsonc")
new_index_df = pd.read_csv(label_config["prl_df"], index_col="subid")

src_root = Path("/media/smbshare/3Tpioneer_bids")
# dst_root = DATA_ROOT
dst_root = Path("/media/smbshare/srs-9/prl_project/full_inference_data")

# TODO update this to look for anything matching a pattern like r"lesion.t3m20/prl_mask_def_prob_([A-Z]+_?)+.nii.gz"
files_to_copy = [
    "t1.nii.gz",
    "phase.nii.gz",
    "flair.nii.gz",
    "lst-ai/lstai_lesion_index.nii.gz",
    "lst-ai/space-flair_seg-lst.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_LR.nii.gz", 
    "lesion.t3m20/prl_mask_def_prob_CH.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_SRS.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_SRS_CH.nii.gz",
]

create_lesion_index_sh = "/home/srs-9/Projects/prl_project/preprocessing/create_lstai_lesion_index.sh"

with open(label_config["subjects"], "r") as f:
    subjects_labeled = [str(line.strip()) for line in f.readlines()]

# subjects = [1082, 1101, 1118, 1126, 1130, 1133, 1152, 1156, 1164, 1165, 1177, 1178, 1183, 1186, 1201, 1209]
with open("/home/srs-9/Projects/prl_project/notebooks/unlabeled_subids.txt", 'r') as f:
    subjects_unlabeled = [line.strip() for line in f.readlines()]

full_df = pd.read_csv("/home/srs-9/Projects/thalamus_project/data/subject-sessions-updated.csv", index_col="sub")

all_subs = []
# for subid, row in new_index_df.iterrows():
for subid in tqdm(full_df.index):
    if str(subid) in subjects_labeled or str(subid) in subjects_unlabeled:
        continue
    subid = int(subid)
    row = full_df.loc[subid, :]
    sesid = row['ses']
    src_dir = src_root / f"sub-ms{subid}/ses-{sesid}"
    dst_dir = dst_root / f"sub{subid}-{sesid}"
    if not (src_dir / "flair.nii.gz").exists() or not (src_dir / "phase.nii.gz").exists():
        continue
    dst_dir.mkdir(exist_ok=True)
    all_subs.append(subid)
    continue
    for file in files_to_copy:
        src_path = src_dir / file
        dst_path = dst_dir / src_path.name

        if src_path.name == "lstai_lesion_index.nii.gz" and not src_path.exists():
            subprocess.run(f"bash {create_lesion_index_sh} {subid} {sesid}", text=True, shell=True)
        if dst_path.exists():
            continue
        if src_path.exists():
            print(f"Copying {src_path} to {dst_path}")
            shutil.copy2(src_path, dst_path)

with open("full_subject.txt", 'w') as f:
    for s in all_subs:
        f.write(str(s) + "\n")