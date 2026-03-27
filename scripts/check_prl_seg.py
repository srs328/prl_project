import pyperclip
from pathlib import Path
import os
import subprocess
import sys
import pandas as pd

sys.path.append("/home/srs-9/Projects/prl_project/helpers")
from shell_interface import open_itksnap_workspace_cmd

label_ref = pd.read_csv("/home/srs-9/Projects/prl_project/src/resources/PRL_spreadsheet-lstai_update_label_reference.csv",
                        index_col="subid")

# dataroot = Path("/media/smbshare/srs-9/prl_project/data")
dataroot = Path(os.environ['PRL_DATA_ROOT'])
subid = int(sys.argv[1])
sesid = label_ref.loc[subid, "date_mri"]

prl_labels = [
    "prl_mask_def_prob_CH.nii.gz",
    "prl_mask_def_prob_SRS.nii.gz",
    "prl_mask_def_prob_LR.nii.gz",
    "prl_mask_def_prob_SRS_CH.nii.gz",
]
images = ["flair.nii.gz", "phase.nii.gz"]

subject_root = dataroot / f"sub{subid}-{sesid}"
for lab in prl_labels.copy():
    if not (subject_root / lab).exists():
        prl_labels.remove(lab)

print("Existing labels:")
print("\n".join(prl_labels))
print("\n")

print(f"Total PRL: {label_ref.loc[subid, "Total PRL"]}")
for i in range(20):
    lab_col = f"PRL{i+1}_label"
    conf_col = f"confidence.{i}"
    
    if not pd.isna(label_ref.loc[subid, lab_col]):
        print(f"{lab_col}: {label_ref.loc[subid, lab_col]}, "
              f"confidence: {label_ref.loc[subid, conf_col]}")
        
image_paths = [subject_root / im for im in images]
label_paths = [subject_root / lab for lab in prl_labels+["lstai_lesion_index.nii.gz"]]

rename_root=("/mnt/z", "Z:/")
# rename_root = None
cmd = open_itksnap_workspace_cmd(image_paths, labels=label_paths, rename_root=rename_root)
# cmd = open_itksnap_workspace_cmd(image_paths, labels=label_paths)
print(cmd)
pyperclip.copy(cmd)
# subprocess.run(cmd, shell=True)