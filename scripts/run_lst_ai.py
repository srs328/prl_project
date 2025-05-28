import subprocess
from pathlib import Path
import json

with open("/home/srs-9/Projects/prl_project/data/subject-sessions-longit.json", 'r') as f:
    subject_sessions = json.load(f)

with open("/home/srs-9/Projects/prl_project/subjects_to_process.txt", 'r') as f:
    subjects = [line.strip() for line in f.readlines()]

dataroot = Path("/media/smbshare/3Tpioneer_bids")
lst_ai_script = "/home/srs-9/Projects/prl_project/run_lst_ai.sh"

for subid in subjects:
    sessions = sorted(subject_sessions[subid])
    work_dir = dataroot / f"sub-ms{subid}" / f"ses-{sessions[0]}"
    cmd = ["bash", lst_ai_script, work_dir]
    print([str(item) for item in cmd])
    subprocess.run(cmd)