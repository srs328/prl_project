# %%
import os
import sys
import argparse
from pathlib import Path
from monai.apps.auto3dseg import AutoRunner
from monai.config import print_config
import shutil
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers.paths import load_config

print_config()

# %%
def make_argument_parser(argv):
    parser = argparse.ArgumentParser(
        description="Train MONAI Auto3DSeg model"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to run directory. If not provided, auto-increments to next available run<N>"
    )
    return parser

# %%
label_config = load_config("label_config.json")
monai_config = load_config("monai_config.json")

# %%
# Parse command-line arguments
parser = make_argument_parser(sys.argv)
args = parser.parse_args()

dataroot = label_config['dataroot']
train_home = Path(label_config['train_home'])
expand_xy = label_config['expand_xy']
expand_z = label_config['expand_z']

training_work_home = Path(monai_config['training_work_home'])

# Determine work_dir: either from --run-dir arg or auto-increment
if args.run_dir:
    work_dir = Path(args.run_dir)
else:
    # Auto-increment to next available run<N>
    run_num = 1
    while (training_work_home / f"run{run_num}").exists():
        run_num += 1
    work_dir = training_work_home / f"run{run_num}"

datalist_file_src = train_home / f"datalist_xy{expand_xy}_z{expand_z}.json"
datalist_file = work_dir / f"datalist_xy{expand_xy}_z{expand_z}.json"

train_param = monai_config['train_param']
algos = train_param.pop("algos")

description = f"""Training run
expand_xy={expand_xy}, expand_z={expand_z}
work_dir={work_dir}
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

# shutil.copyfile(datalist_file_src, datalist_file)
# shutil.copyfile("monai_config.json", work_dir/"monai_config.json")
# shutil.copyfile("label_config.json", work_dir/"label_config.json")


with open(work_dir / "info.txt", 'w') as f:
    f.write(description)

print("work_dir is: ", work_dir)

# %%
# Set up MLflow for unified cross-fold tracking
mlflow_tracking_uri = str(work_dir / "mlruns")
mlflow_experiment_name = f"run{work_dir.name[3:]}" if work_dir.name.startswith("run") else work_dir.name

# List-valued params in train_param get mangled by BundleAlgo.train()'s CLI
# conversion (Fire interprets '1,1,4' as a string, not [1,1,4]). Moving them
# to the input dict ensures they flow through fill_template_config into
# hyper_parameters.yaml correctly without a CLI override step.
input_dict = {
    "modality": "MRI",
    "datalist": str(datalist_file),
    "dataroot": str(dataroot),
}
for key in list(train_param):
    if isinstance(train_param[key], list):
        input_dict[key] = train_param.pop(key)

runner = AutoRunner(
    work_dir=work_dir,
    algos=algos,
    input=input_dict,
    mlflow_tracking_uri=mlflow_tracking_uri,
    mlflow_experiment_name=mlflow_experiment_name,
)
runner.set_training_params(train_param)

# %%

runner.run()