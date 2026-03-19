# %%
import os
from pathlib import Path
from monai.apps.auto3dseg import AutoRunner
from monai.config import print_config
import shutil
import json

print_config()

# %%
dataroot = Path("/media/smbshare/srs-9/prl_project/data")

with open("label_config.json", 'r') as f:
    label_config = json.load(f)
    
with open("monai_config.json", 'r') as f:
    monai_config = json.load(f)

dataroot = label_config['dataroot']
train_home = Path(label_config['train_home'])
expand_xy = label_config['expand_xy']
expand_z = label_config['expand_z']

training_work_home = Path(monai_config['training_work_dir'])
work_dir = training_work_home / "run2" #! edit this
datalist_file_src = train_home / f"datalist_xy{expand_xy}_z{expand_z}.json"
datalist_file = work_dir / f"datalist_xy{expand_xy}_z{expand_z}.json"

train_param = monai_config['train_param']
algos = train_param["algos"]


description = """Info
"""

# %%
with open(datalist_file_src, 'r') as f:
    datalist = json.load(f)

for item in datalist['training']:
    assert os.path.exists(item['image'])
    assert os.path.exists(item['label'])

for item in datalist['testing']:
    assert os.path.exists(item['image'])
    
# %%
if not work_dir.exists():
    print(f"Creating {str(work_dir)}")
    os.makedirs(work_dir)

shutil.copyfile(datalist_file_src, datalist_file)
shutil.copyfile("monai_config.json", work_dir/"monai_config.json")
shutil.copyfile("label_config.json", work_dir/"label_config.json")


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
runner.set_training_params(train_param)

# %%

runner.run()