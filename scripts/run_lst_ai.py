import subprocess
from pathlib import Path
import json

with open("/home/srs-9/Projects/prl_project/data/subject-sessions-longit.json", 'r') as f:
    subject_sessions = json.load(f)

with open("/home/srs-9/Projects/prl_project/subjects_to_process.txt", 'r') as f:
    subjects = [int(line.strip()) for line in f.readlines()]

subjects.sort()

dataroot = Path("/media/smbshare/3Tpioneer_bids")
lst_ai_script = "/home/srs-9/Projects/prl_project/scripts/run_lst_ai.sh"

print("Starting")
for subid in subjects:
    subid = str(subid)
    sessions = sorted(subject_sessions[subid])
    work_dir = dataroot / f"sub-ms{subid}" / f"ses-{sessions[0]}"
    cmd = ["bash", lst_ai_script, work_dir]
    print([str(item) for item in cmd])
    subprocess.run(cmd)