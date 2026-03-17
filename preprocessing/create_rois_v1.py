"""
The version used in roi_train1. The issue here was that if a lesion had a prl, it wouldn't even look at the lst-ai lesion segmentation, so there were cases (e.g. sub1038) where big parts of large lesions were missing from the training
"""
# %%
import sys
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append("/home/srs-9/Projects/prl_project")
from helpers.parallel import BetterPool
from preprocessing.verify_segmentations import verify_prl, verify_lesion

PROCESSES = 12
EXPAND_XY = 20
EXPAND_Z = 2
DEFINE_BOXES_SH = "/home/srs-9/Projects/prl_project/preprocessing/define_bounding_boxes.sh"

images = [
    "flair.nii.gz",
    "phase.nii.gz",
    "t1.nii.gz",
]
segmentations = [
    "prl_mask_def_prob_CH.nii.gz",
    "prl_mask_def_prob_SRS.nii.gz",
    "prl_mask_def_prob_LR.nii.gz",
]
lesion_mask = "space-flair_seg-lst.nii.gz"

#%% 
def prepare_segmentation(seg_path):
    command(
        f"fslmaths {seg_path} -uthr 2 -thr 2 {seg_path.parent}/rim.nii.gz"
    )
    command(
        f"fslmaths {seg_path} -uthr 1 -thr 1 {seg_path.parent}/core.nii.gz"
    )
    command(
        f"fslmaths {seg_path} -thr 3 -bin -add {seg_path.parent}/core.nii.gz {seg_path.parent}/core.nii.gz"
    )
    command(
        f"fslmaths {seg_path.parent}/core.nii.gz -add {seg_path.parent}/rim.nii.gz -uthr 3 {seg_path.parent}/prl_label_final.nii.gz"
    )
    if not verify_prl(f"{seg_path.parent}/prl_label_final.nii.gz"):
        raise Exception

from helpers.shell_interface import command

# subject_list = sys.argv[1]
subject_list = "/home/srs-9/Projects/prl_project/training/roi_train1/subjects.txt"

dataroot = Path("/media/smbshare/srs-9/prl_project/data")
with open(subject_list, 'r') as f:
    subjects = [int(line.strip()) for line in f.readlines()]
    
prl_df = pd.read_csv(
    "/home/srs-9/Projects/prl_project/PRL_spreadsheet-lstai_update_label_reference.csv",
    index_col="subid"
)

def confidence_label(i):
    if i == 1:
        return "confidence"
    else:
        return f"confidence.{i-1}"
    
    

def prepare_rois(subid):
    sesid = prl_df.loc[subid, "date_mri"]
    subject_root = dataroot / f"sub{subid}-{sesid}"
    bounding_box_file = subject_root / f"lstai_bounding_boxes_xy{EXPAND_XY}_z{EXPAND_Z}.txt"
    if not bounding_box_file.exists():
        command(
            f"bash {DEFINE_BOXES_SH} --expand-xy {EXPAND_XY} --expand-z {EXPAND_Z} {subject_root}"
        )

    with open(bounding_box_file, 'r') as f:
        bounding_boxes = []
        for line in f.readlines():
            parts = line.split(" ")
            bounding_boxes.append((parts[0], " ".join(parts[1:]).strip()))

        
    prl_labels = [int(prl_df.loc[subid, f"PRL{i}_label"]) for i in range(1,21) 
                if prl_df.loc[subid,confidence_label(i)] in ["definite", "probable"]]

    # %%

    prl_to_verify = []
    for index, box in tqdm(bounding_boxes, total=len(bounding_boxes)):
        output_path = subject_root / str(index)
        output_path.mkdir(exist_ok=True)
        if int(index) in prl_labels:
            for seg in segmentations:
                if (subject_root/seg).exists():
                    label_in = subject_root/seg
                    label_out = output_path/"prl_label.nii.gz"
                    prl_to_verify.append(label_out)
                    cmd = f"fslroi {label_in} {label_out} {box}"
                    # print(cmd)
                    # command(cmd)
                    break
            else:
                #! will need to continue the big upper loop, or just preempt this before looping through bounding box
                print(f"No PRL segmentation found for {subid}")

        label_in = subject_root / lesion_mask 
        label_out = output_path / "lesion.nii.gz"   
        cmd = f"fslroi {label_in} {label_out} {box}"
        print(cmd)
        command(cmd)
        for input_name in images:
            cmd = f"fslroi {subject_root/input_name} {output_path/input_name} {box}"
            print(cmd)
            command(cmd)


    for prl_label in prl_to_verify:
        if not prl_label.exists():
            continue
        result = command(
            f"fslstats {prl_label} -R"
        ).stdout
        max_lab = int(float(result.split(" ")[1].strip()))
        if max_lab < 2:
            print(f"Missing segmentation for sub{subid} lesion {index}")
            command(f"rm {prl_label}")
            continue
        prepare_segmentation(prl_label)

    # "fslroi [INPUT_IMAGE] $SUBJECTDIR/prlmontage/$sessionDate/prl${lesion_label}.phase.box.nii.gz ${roi_boundaries}"

# pool = BetterPool(PROCESSES)
# pool.map(prepare_rois, subjects)
for subid in subjects:
    prepare_rois(subid)
# prepare_rois(2041)