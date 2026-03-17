from pathlib import Path
import os
import shutil
import pandas as pd

subject_sessions = pd.read_csv("/home/srs-9/Projects/prl_project/data/subject-sessions.csv",
                               index_col="sub")

new_index_df = pd.read_csv("/home/srs-9/Projects/prl_project/PRL_spreadsheet-lstai_update_label_reference.csv",
                        index_col="subid")

src_root = Path("/media/smbshare/3Tpioneer_bids")
dst_root = Path("/media/smbshare/srs-9/prl_project/data")

files_to_copy = [
    "t1.nii.gz",
    "phase.nii.gz",
    "flair.nii.gz",
    "lst-ai/lstai_lesion_index.nii.gz",
    "lst-ai/space-flair_seg-lst.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_LR.nii.gz", 
    "lesion.t3m20/prl_mask_def_prob_CH.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_SRS.nii.gz",
]

for subid, row in new_index_df.iterrows():
    sesid = row['date_mri']
    src_dir = src_root / f"sub-ms{subid}/ses-{sesid}"
    dst_dir = dst_root / f"sub{subid}-{sesid}"
    dst_dir.mkdir(exist_ok=True)
    for file in files_to_copy:
        src_path = src_dir / file
        dst_path = dst_dir / src_path.name
        if dst_path.exists():
            continue
        if src_path.exists():
            print(f"Copying {src_path} to {dst_path}")
            shutil.copy2(src_path, dst_path)
