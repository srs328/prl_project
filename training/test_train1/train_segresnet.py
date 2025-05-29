# %%
import os
from pathlib import Path
from monai.apps.auto3dseg import AutoRunner
from monai.config import print_config
import shutil

print_config()

# %%
dataroot = Path("/media/smbshare/3Tpioneer_bids")
training_work_home = Path("/home/srs-9/Projects/prl_project/training_work_dirs")
work_dir = training_work_home / "test_train1_segresnet" #! edit this
datalist_file_src = os.path.join(os.getcwd(), "datalist.json")
datalist_file = work_dir / "datalist.json"
algos = ["segresnet"]

description = """Model: SegResNet
Trained using flair.phase images. 
Labels created by combining definite and probably PRLs and adding those to the original LST lesion segmentations"""

# %% [markdown]
# Check that every file in the datalist exists

# %%
import json

with open(datalist_file_src, 'r') as f:
    datalist = json.load(f)

for item in datalist['training']:
    assert os.path.exists(item['image'])
    assert os.path.exists(item['label'])

for item in datalist['testing']:
    assert os.path.exists(item['image'])
    assert os.path.exists(item['label'])

# %% [markdown]
# Create the work_dir, copy over the datalist, and save an info file

# %%
if not work_dir.exists():
    print(f"Creating {str(work_dir)}")
    os.makedirs(work_dir)

shutil.copyfile(datalist_file_src, datalist_file)

with open(work_dir / "info.txt", 'w') as f:
    f.write(description)

print("work_dir is: ", work_dir)

# %%
runner = AutoRunner(
    work_dir=work_dir,
    algos=algos,
    input={
        "modality": "MRI",
        "datalist": str(datalist_file),
        "dataroot": str(dataroot),
    },
)

# %%
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


