import subprocess
import os
import pandas as pd
from Pathlib import Path

orig_index_df = pd.read_csv("/home/srs-9/Projects/prl_project/PRL_spreadsheet_label_reference.csv",
                         index_col="subid")
new_index_df = pd.read_csv("/home/srs-9/Projects/prl_project/PRL_spreadsheet-lstai_update_label_reference.csv",
                        index_col="subid")
dataroot = Path("/mnt/h/3Tpioneer_bids")
work_dir = Path(__file__).parent / "tmp"
work_dir.mkdir(exist_ok=True)
mask = work_dir / "mask.nii.gz"
masked = work_dir / "masked.nii.gz"

prl_labels = [
    "lesion.t3m20/prl_mask_def_prob_LR.nii.gz" 
    "lesion.t3m20/prl_mask_def_prob_CH.nii.gz"
    "lesion.t3m20/prl_mask_def_prob_SRS.nii.gz"
]

orig_index_file = "lesion.t3m20/lesion_index.t3m20.nii.gz"
new_index_file = "lst-ai/lstai_lesion_index.nii.gz"

for subid, row in orig_index_df.iterrows():
    sesid = row['date_mri']
    subject_root = dataroot / f"sub-ms{subid}/ses-{sesid}"
    orig_index_map = subject_root / orig_index_file
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
        prl_index = row[f"PRL_LABEL{i}"]
        cmd = f"fslmaths {orig_index_map} -uthr {prl_index} -thr {prl_index} -bin {work_dir}/mask.nii.gz"
        subprocess.run(cmd, shell=True)
        cmd = f"fslmaths {new_index_map} -mul {mask} {masked}"
        subprocess.run(cmd, shell=True)
        cmd = f"fslstats {masked} -R"