import json
import os
import re
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from loguru import logger

from helpers.paths import load_config
from helpers.shell_interface import command, run_if_missing
from core.dataset import Dataset
from core.experiment import Experiment

from analysis.image import classify


## Inputs: 

dataset_name = "roi_train2"
dataset = Dataset(dataset_name)

experiment_key, run_name = "stage6", "run1"
experiment_name = f"{EXPERIMENT_KEYS[experiment_key]}/{run_name}"
experiment = Experiment.from_run_dir(experiment_name, dataset)

expand_xy: int = experiment.preprocess_config.expand_xy
expand_z: int = experiment.preprocess_config.expand_z
images: tuple[str, ...] = experiment.preprocess_config.images

inference_dataset = Dataset("inference_dataset")
inference_dataset.create_datalist()