"""Standalone script to run SegResNet2D training in an existing run directory.

Exploratory — hardcoded to run3. Calls AutoRunner with algos=["segresnet2d"]
so it creates segresnet2d_0–4 folds alongside the existing segresnet_0–4.

Usage:
    python training/roi_train2/train_2d.py
"""

import logging
import os
import sys

os.environ["SEGRESNET2D_ALWAYS"] = "1"

from monai.apps.auto3dseg import AutoRunner
from monai.config import print_config

print_config()
logging.getLogger("monai.apps.auto3dseg").setLevel(logging.DEBUG)

WORK_DIR = "/media/smbshare/srs-9/prl_project/training/roi_train2/run3"
DATALIST = f"{WORK_DIR}/datalist_flair.phase_xy20_z2.json"
DATAROOT = "/media/smbshare/srs-9/prl_project/data"

input_dict = {
    "modality": "MRI",
    "datalist": DATALIST,
    "dataroot": DATAROOT,
    "learning_rate": 0.0002,
    "num_images_per_batch": 1,
    "num_epochs": 500,
    "num_warmup_epochs": 1,
    "num_epochs_per_validation": 1,
    "roi_size": [44, 44, 8],
}

# runner = AutoRunner(
#     work_dir=WORK_DIR,
#     algos=["segresnet2d"],
#     input=input_dict,
#     analyze=False,
#     algo_gen=False,
#     train=False,
#     ensemble=True,
#     mlflow_tracking_uri=f"{WORK_DIR}/mlruns",
#     mlflow_experiment_name="3",
# )

# runner.run()
