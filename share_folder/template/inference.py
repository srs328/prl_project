# %%
import json
import os
from pathlib import Path
import shutil

from monai.apps.auto3dseg import (
    AlgoEnsembleBestN,
    AlgoEnsembleBuilder,
    import_bundle_algo_history,
)

# %%
model = "segresnet" 
work_dir_name = f"{model}_work_dir" # whatever you named the work_dir during training

# this will be where inference gets saved
savedir_name = f"{model}_inference" 

task_name = "" # whatever you want, or leave blank

# %%
dataroot = Path("/media/smbshare/3Tpioneer_bids")
curr_path = os.path.realpath(__file__)
curr_dir = Path(os.path.dirname(curr_path))

save_dir = curr_dir / savedir_name
network_bundle_path = curr_dir / work_dir_name
datalist_file_src = curr_dir / "inference_datalist.json"
datalist_file = save_dir / "inference_datalist.json"

shutil.copyfile(datalist_file_src, datalist_file)

# you can customize the way it saves predictions, but don't worry about it for now
save_params = {
    "_target_": "SaveImage",
    "output_dir": save_dir,
    "data_root_dir": dataroot,
    "output_postfix": "",
    "separate_folder": True
}

# %%
task = {
    "name": task_name,
    "task": "segmentation",
    "modality": "MRI",
    "datalist": datalist_file,
    "dataroot": dataroot,
}

task_file = os.path.join(save_dir, "inference-task.json")
with open(task_file, "w") as f:
    json.dump(task, f, indent=4)

input_cfg = task_file  # path to the task input YAML file created by the users
history = import_bundle_algo_history(network_bundle_path, only_trained=True)

## model ensemble
n_best = 5
builder = AlgoEnsembleBuilder(history, input_cfg)
builder.set_ensemble_method(AlgoEnsembleBestN(n_best=n_best))
ensemble = builder.get_ensemble()
ensemble(pred_param={"image_save_func": save_params})


