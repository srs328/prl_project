#%%
from pathlib import Path
import json
import os
import shutil
import subprocess
from loguru import logger
import sys
from tqdm import tqdm

# from helpers.paths import TRAIN_ROOT
from core.dataset import Dataset

curr_file = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_file)

# Custom sink that uses tqdm.write
class TqdmLoguruStream:
    def write(self, message):
        tqdm.write(message, end='')
    
    def flush(self):
        pass

logger.remove()  # Remove the default handler
logger.add(TqdmLoguruStream(), level="INFO", format="{time:YYYY-MM-DD HH:MM:SS} | {level} | {message}")
# logger.add(sys.stderr, level="INFO")  # Add a new handler with WARNING level
logger.add(os.path.join(curr_dir, "run_lst_ai.log"), level="DEBUG")
lstai_script = "/home/srs-9/Projects/prl_project/src/scripts/lst_ai.sh"

#%%
# dataset_name = "inference_dataset"
dataset_name = "roi_train2"
dataset = Dataset(dataset_name)
# work_home = Path("/media/smbshare/srs-9/fastsurfer")
# dataroot = Path("/media/smbshare/3Tpioneer_bids")
# dataroot = Path("")

subjects = dataset.subjects
# for subid in subject_sessions:
for subid in tqdm(subjects[:2], total=len(subjects), desc="Processing subjects", unit="subject"):
    logger.info(f"Starting subject {subid}")
    subject = dataset.subject(subid)

    work_dir = subject.dir

    cmd = ["bash", lstai_script, str(work_dir)]
    cmd_str = " ".join([str(item) for item in cmd])
    logger.info(cmd_str)
    try:    
        result = subprocess.run(cmd, text=True, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        tqdm.write(f"ERROR: {subid} failed")
        logger.error(e.stderr)
        continue
    else:
        logger.debug(result.stdout)
    