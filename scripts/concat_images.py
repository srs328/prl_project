from pathlib import Path
import json
import subprocess
import sys

subjects_file = sys.argv[1]

with open("/home/srs-9/Projects/prl_project/data/subject-sessions-longit.json", 'r') as f:
    subject_sessions = json.load(f)

with open(subjects_file, 'r') as f:
    subjects = [line.strip() for line in f.readlines()]

dataroot = Path("/media/smbshare/3Tpioneer_bids")
script = "/home/srs-9/Projects/prl_project/scripts/concatImages.sh"

with open("errors.log", 'w') as f:
    for subid in subjects:
        sessions = sorted(subject_sessions[subid])
        work_dir = dataroot / f"sub-ms{subid}" / f"ses-{sessions[0]}"
        cmd = ["bash", script, work_dir]
        print([str(item) for item in cmd])

        try:
            result = subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(subid)
            print("Stderr:", e.stderr)
            f.write(subid + "\n" + str(e.stderr) + "\n\n")
            continue

