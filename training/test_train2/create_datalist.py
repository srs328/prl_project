from pathlib import Path
import json

subjects = []
with open("subjects.txt", 'r') as f:
    subjects = [line.strip() for line in f.readlines()]

dataroot = Path("/media/smbshare/srs-9/prl_project/training/test_train2")

datalist = {"testing": [], "training": []}
for sub in subjects:
    im = dataroot / "imagesTr" / f"sub{sub}.nii.gz"
    lab = dataroot / "labelsTr" / f"sub{sub}_2.nii.gz"
    datalist["training"].append({"image": str(im), "label": str(lab)})

with open("datalist.json", 'w') as f:
    json.dump(datalist, f, indent=4)