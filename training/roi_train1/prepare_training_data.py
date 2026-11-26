import re
import sys
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append("/home/srs-9/Projects/prl_project")
from preprocessing.verify_segmentations import verify_prl, verify_lesion
from helpers.shell_interface import make_screenshot, command

curr_dir = Path(__file__).parent
screenshots = curr_dir / "screenshots"
CONCAT_SH = "/home/srs-9/Projects/prl_project/scripts/concatImages.sh"

subject_list = "/home/srs-9/Projects/prl_project/training/roi_train1/subjects.txt"

dataroot = Path("/media/smbshare/srs-9/prl_project/data")
with open(subject_list, 'r') as f:
    subjects = [int(line.strip()) for line in f.readlines()]

prl_df = pd.read_csv(
    "/home/srs-9/Projects/prl_project/PRL_spreadsheet-lstai_update_label_reference.csv",
    index_col="subid"
)

all_prl_labels = []
all_lesion_labels = []
for subid in subjects:
    subid = int(subid)
    sesid = prl_df.loc[subid, "date_mri"]
    subject_root = dataroot / f"sub{subid}-{sesid}"

    lesion_folders = [Path(item.path) for item in os.scandir(subject_root) if item.is_dir() and re.match(r"^\d+", item.name)]
    for folder in lesion_folders:
        image = folder / "flair.phase.nii.gz"
        if not image.exists():
            command(f"bash {CONCAT_SH} {folder} {image} flair.nii.gz phase.nii.gz")
        prl_label = folder / "prl_label_final.nii.gz"
        if prl_label.exists():
            if verify_prl(prl_label):
                print("GOOD: ", prl_label)
                img_path = folder / "phase.nii.gz"
                seg_path = prl_label
                # make_screenshot(img_path, seg_path, 
                #                 screenshots/f"sub{subid}_PRL_ind{folder.name}.png", colormap="red-yellow", alpha=0.7)
                all_prl_labels.append((image, seg_path))
                continue
            else:
                # print("BAD: ", prl_label)
                pass
        lesion = folder / "lesion.nii.gz"
        if not verify_lesion(lesion):
            print("BAD: ", lesion)
            continue
        all_lesion_labels.append((image, lesion))
                
with open(curr_dir/"prl_data.txt", 'w') as f:
    f.write("image,label\n")
    for line in all_prl_labels:
        f.write(f"{line[0]},{line[1]}\n")
            
with open(curr_dir/"lesion_data.txt", 'w') as f:
    f.write("image,label\n")
    for line in all_lesion_labels:
        f.write(f"{line[0]},{line[1]}\n")
            