import subprocess
from pathlib import Path
import json
import os
import csv

curr_path = os.path.realpath(__file__)
curr_dir = os.path.dirname(curr_path)
os.chdir(curr_dir)

with open("/home/srs-9/Projects/prl_project/data/subject-sessions.csv", 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    subject_sessions = {row[0]: row[1] for row in reader}


with open("/home/srs-9/Projects/prl_project/subject_lists/green_subjects_w_prl_labels.txt", 'r') as f:
    subjects = [line.strip() for line in f.readlines()]


dataroot = Path("/media/smbshare/3Tpioneer_bids")
script = "createLabels.sh"

with open("errors.log", 'w') as f:
    for subid in subjects:
        sesid = subject_sessions[subid]
        work_dir = dataroot / f"sub-ms{subid}" / f"ses-{sesid}"
        cmd = ["bash", script, work_dir]
        print([str(item) for item in cmd])

        try:
            result = subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(subid)
            print("Stderr:", e.stderr)
            f.write(subid + "\n" + str(e.stderr) + "\n\n")
            continue

