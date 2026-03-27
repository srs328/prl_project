import pandas as pd
from pathlib import Path
import os
import sys
import pyperclip

from mri_data import utils

subject_sessions = pd.read_csv("/home/srs-9/Projects/prl_project/data/subject-sessions.csv",
                               index_col="sub")
dataroot = Path("/mnt/h/3Tpioneer_bids")

subid = sys.argv[1]
sesid = subject_sessions.loc[int(subid), 'ses']
session_root = dataroot / f"sub-ms{subid}/ses-{sesid}"

images = ["t1.nii.gz", "flair.nii.gz", "phase.nii.gz"]
labels = [
    "lesion.t3m20/lesion_index.t3m20.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_LR.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_CH.nii.gz",
    "lesion.t3m20/prl_mask_def_prob_SRS.nii.gz",
    "lst-ai/space-flair_seg-lst.nii.gz",
    "lst-ai/lstai_lesion_index.nii.gz",
]

image_paths = [session_root/im for im in images if (session_root/im).exists()]
label_paths = [session_root/lab for lab in labels if (session_root/lab).exists()]

cmd = utils.open_itksnap_workspace_cmd(image_paths, label_paths, win=True)
pyperclip.copy(cmd)
print(cmd)