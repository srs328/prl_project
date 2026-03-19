from pathlib import Path
import os
import shutil
import pandas as pd
import subprocess

subject_sessions = pd.read_csv("/home/srs-9/Projects/prl_project/resources/subject-sessions.csv",
                               index_col="sub")

new_index_df = pd.read_csv("/home/srs-9/Projects/prl_project/PRL_spreadsheet-lstai_update_label_reference.csv",
                        index_col="subid")

dst_root = Path("/media/smbshare/3Tpioneer_bids")
src_root = Path("/media/smbshare/srs-9/prl_project/data")

prl_labels = [
    "prl_mask_def_prob_CH.nii.gz",
    "prl_mask_def_prob_SRS.nii.gz",
    "prl_mask_def_prob_LR.nii.gz",
]

cmds = []
for subid, row in subject_sessions.iterrows():
    ses = row['ses']
    subject_root = src_root / f"sub{subid}-{ses}"
    dst_folder = dst_root / f"sub-ms{subid}/ses-{ses}/lesion.t3m20"

    for lab in prl_labels:
        src = subject_root / lab
        dst = dst_folder / lab
        if src.exists():
            cmd = f"rsync -avhu {src} {dst}"
            print(cmd)
            # subprocess.run(f"rsync -avhu {src} {dst}")
            cmds.append(cmd)

with open("copy_commands.sh", 'w') as f:
    f.writelines("\n".join(cmds))