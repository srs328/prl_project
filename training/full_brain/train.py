# %%
from monai.apps.auto3dseg import AutoRunner
import json
import shutil
from pathlib import Path
import os

from helpers.paths import DATA_ROOT

# %%
train_home = Path("/home/srs-9/Projects/prl_project/training/full_brain")
train_root = Path("/media/smbshare/srs-9/prl_project/training/full_brain")
train_name = "swinunetr_test1"
work_dir = train_root / train_name
work_dir.mkdir(exist_ok=True)

mlflow_tracking_uri = str(work_dir / "mlruns")
run_name = train_name
mlflow_experiment_name = (
    run_name[3:] if run_name.startswith("run") else run_name
)

datalist_dst = work_dir/"datalist.json"
shutil.copy2(train_home/"datalist.json", datalist_dst)

# input_cfg = {
#     "modality": "mri",
#     "class_names": ["lesion", "rim"],
#     "algos": ["swinunetr"],
#     "work_dir": str(work_dir),
#     "datalist": str(datalist_dst),
#     "dataroot": str(DATA_ROOT),
#     "analyze": True,
#     "algo_gen": True,
#     "train": False,
#     "ensemble": False
# }

input_overrides = "/home/srs-9/Projects/prl_project/training/full_brain/input_overrides.yaml"

# train_params = {
#     "num_epochs": 500
# }
runner = AutoRunner(input=input_overrides,
                    analyze=True,
                    algo_gen=True,
                    train=False,
                    ensemble=False,
                    mlflow_tracking_uri=mlflow_tracking_uri,
                    mlflow_experiment_name=mlflow_experiment_name,
)
runner.run()

# %%
from monai.bundle import ConfigParser


def patch_swinunetr_configs(work_dir, overrides, num_folds=5, algo_name="swinunetr"):
    """Patch generated config YAMLs with user overrides.

    Uses MONAI's '#' nested-key notation, e.g. "loss#weight" targets config["loss"]["weight"],
    and "transforms_train#transforms#9#ratios" targets the 10th transform's ratios field.
    """
    work_dir = Path(work_dir)
    for fold in range(num_folds):
        config_dir = work_dir / f"{algo_name}_{fold}" / "configs"
        for config_file, patches in overrides.items():
            fpath = config_dir / config_file
            parser = ConfigParser(globals=False)
            parser.read_config(str(fpath))
            for key, value in patches.items():
                parser[key] = value
            ConfigParser.export_config_file(
                parser.get(), str(fpath), fmt="yaml", default_flow_style=None
            )


overrides = {
    "hyper_parameters.yaml": {
        "loss#weight": "$torch.tensor([1.0, 1.0, 10.0]).cuda()",
    },
    "transforms_train.yaml": {
        "transforms_train#transforms#9#ratios": [0, 1, 5],
    },
}
patch_swinunetr_configs(work_dir, overrides)

# %%
# Verify overrides took effect
p = ConfigParser(globals=False)
p.read_config(str(work_dir / "swinunetr_0/configs/hyper_parameters.yaml"))
print("loss:", p["loss"])

p2 = ConfigParser(globals=False)
p2.read_config(str(work_dir / "swinunetr_0/configs/transforms_train.yaml"))
crop_transform = p2["transforms_train#transforms#9"]
print("RandCropByLabelClassesd:", crop_transform)

# %%
# Train with patched configs (skips analyze + algo_gen since bundles already exist)
runner2 = AutoRunner(
    work_dir=str(work_dir),
    input=input_overrides,
    analyze=False,
    algo_gen=False,
    train=True,
    ensemble=True,
    mlflow_tracking_uri=mlflow_tracking_uri,
    mlflow_experiment_name=mlflow_experiment_name,
)
runner2.run()

# # %%
# import yaml
# train_home = Path("/home/srs-9/Projects/prl_project/training/roi_train2")
# train_root = Path("/media/smbshare/srs-9/prl_project/training/roi_train2")
# train_name = "tmp2"
# work_dir = train_root / train_name
# work_dir.mkdir(exist_ok=True)

# datalist_dst = work_dir/"datalist.json"
# shutil.copy2(train_home/"datalist_flair.phase_xy25_z2.json", datalist_dst)

# input_cfg = {
#     "modality": "mri",
#     "class_names": ["lesion", "rim"],
#     "algos": ["swinunetr"],
#     "work_dir": str(work_dir),
#     "datalist": str(datalist_dst),
#     "dataroot": str(DATA_ROOT),
#     "analyze": True,
#     "algo_gen": True,
#     "train": False,
#     "ensemble": False
# }

# input_overrides = "/home/srs-9/Projects/prl_project/training/full_brain/input_overrides.yaml"

# with open("/media/smbshare/srs-9/prl_project/training/roi_train2/stage7_expand/run1/monai_config.json", 'r') as f:
#     input_overrides = json.load(f)
# input_overrides = input_overrides['train_param']
# input_overrides['datalist'] = str(datalist_dst)
# input_overrides['dataroot'] = str(DATA_ROOT)
# input_overrides['work_dir'] = str(work_dir)
# input_overrides['modality'] = "mri"
# input_overrides['algos'] = ["segresnet"]

# inp_override_path = work_dir/"input_overrides.yaml"
# with open(inp_override_path, 'w') as f:
#     yaml.dump(input_overrides, f)
# runner = AutoRunner(input=str(inp_override_path), train=False, infer=False, analyze=True, algo_gen=True, ensemble=False)
# runner.run()


