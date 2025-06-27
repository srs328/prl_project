# %%
import os
from pathlib import Path
from monai.apps.auto3dseg import AutoRunner
from monai.config import print_config
import shutil
import json
import warnings

print_config()

#%% You can edit these if you want
model = "segresnet" #! edit this

description = \
"""
Fill this in with whatever information
you want about this training. it will be saved to a text file 
"""

# %% Don't change anything below here

# Setup paths
dataroot = Path("/media/smbshare/3Tpioneer_bids")
curr_path = os.path.realpath(__file__)
curr_dir = Path(os.path.dirname(curr_path))
work_dir = curr_dir / f"{model}_work_dir"

datalist_file_src = curr_dir /  "datalist.json"
datalist_file = work_dir / "datalist.json"

# %% Scans the datalist file to make sure all files exist

with open(datalist_file_src, 'r') as f:
    datalist = json.load(f)

testing = datalist['testing']
training = datalist['training']

for item in training:
    if not os.path.exists(item['image']):
        raise FileNotFoundError(f"Cannot find image {item['image']}, double check training data in datalist.json")
    if not os.path.exists(item['label']):
        raise FileNotFoundError(f"Cannot find label {item['label']}, double check training data in datalist file")
    
for item in testing:
    if not os.path.exists(item['image']):
        raise FileNotFoundError(f"Cannot find image {item['image']}, double check testing data in datalist.json")
    if not os.path.exists(item['label']):
        warnings.warn(f"Cannot find label {item['label']}, double check datalist file. Code will run, but you won't have a ground truth to compare")

# %% Creates working directory for Auto3dSeg
valid_models = ["segresnet", "swinunetr", "dints", "segresnet2d"]
if model not in valid_models:
    raise ValueError(f"The model provided is not valid. It should be one of: {", ".join(valid_models)}")

if not work_dir.exists():
    os.makedirs(work_dir)

if not datalist_file_src.exists():
    raise FileNotFoundError("datalist.json file cannot be found")

shutil.copyfile(datalist_file_src, datalist_file)

with open(work_dir / "info.txt", 'w') as f:
    f.write(description)

# %% Runs the training
runner = AutoRunner(
    work_dir=work_dir,
    algos=[model],
    input={
        "modality": "MRI",
        "datalist": str(datalist_file),
        "dataroot": str(dataroot),
    },
)

max_epochs = 250
train_param = {
    "num_epochs_per_validation": 1,
    #"num_images_per_batch": 2,
    "num_epochs": max_epochs,
    "num_warmup_epochs": 1,
}
runner.set_training_params(train_param)

# %%
runner.run()


