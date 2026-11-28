import os
from pathlib import Path
import re

orig_dataroot = Path("/media/smbshare/3Tpioneer_bids")
dataroot1 = Path("/media/smbshare/srs-9/prl_project/data")
dataroot2 = Path("/media/smbshare/srs-9/prl_project/inference_data")
dataroot3 = Path("/media/smbshare/srs-9/prl_project/full_inference_data")

all_folders = []
for dr in [dataroot1, dataroot2, dataroot3]:
    folders = [f for f in dr.glob(r"sub*") if f.is_dir() and f.name.startswith("sub")]
    all_folders.extend(folders)
print(folders)
lines = ""
for folder in all_folders:
    if not folder.is_dir():
        continue
    
    src = folder / "lst-ai"
    match = re.match(r"sub(?P<subid>\d{4})-(?P<sesid>\d{8})", folder.name)
    try:
        subid = match['subid']
        sesid = match['sesid']
    except Exception:
        print(f"Couldn't match {folder}")
        continue

    dst = orig_dataroot / f"sub-ms{subid}/ses-{sesid}"
    try:
        assert dst.exists()
        assert src.exists()
    except AssertionError:
        print("Something doesnt exist")
        continue

    cmd = f"rsync -avhu --ignore-existing {str(src)} {str(dst)}"
    lines += cmd + "\n"

with open("lstai_rsync.sh",  'w') as f:
    f.write(lines)