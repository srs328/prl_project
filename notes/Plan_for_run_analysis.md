
im currently rrunning this block from the notebook "evaluate_gridsearch.ipynb" on all my training stages. The line "exp.cases" takes a bit of time, and even though the Experiment class is written to cache the cases, if I have to restart my notebook session or do this in a script, itd get frustrating. What's a good, simple, effective, and idiomatic way to cache something like this so that i can get load it all up quickly between sessions? like picking into pycache folders? idk actually, so can you come up with something and write it into another file? 

```python
dataset_name = "roi_train2"
for gridsearch_name in gridsearch_experiments:
    # gridsearch_name = "stage5_sweep_dicecewt_nbatch"
    grid = load_grid(dataset_name, gridsearch_name)
    run_params = grid.run_params(*grid.get_info())
    ds = Dataset(dataset_name)
    runs_to_check = ["run1"]
    runs = []
    for run_info in run_params:
        run_loc = f"{grid.experiment_name}/{run_info['run_name']}"
        print(f"STARTING {run_loc}")
        exp = Experiment.from_run_dir(run_loc, ds)

        #Dont rereun this, itll start producing logs and checking whether inference exists 
        #exp.predict() 

        #TODO This is the part I would like to be able to cache
        exp.cases #builds the case list for the first time and caches for quick access
        runs.append({**run_info, "exp": exp})

# clear_output()
```

I do plan on turning a lot of the logic in this notebook into a regular .py module(s), so perhaps you can help with both. 

Originally I was computing metrics for one stage at a time with code like the following:

```python
import core.dataset
import core.experiment
import core.grid
import numpy as np
from scripts.analyze_mlflow_runs import (
    analyze_unified_mlruns,
    aggregate_metrics,
    print_summary,
    plot_metrics,
)
from helpers import utils
from scripts import diagnose_fold_differences as fold_dx
import re
import os
from collections import defaultdict
from reload_recursive import reload_recursive
from pathlib import Path
from IPython.display import clear_output
from tqdm.notebook import tqdm
import pandas as pd
from core.dataset import Dataset
from core.experiment import Experiment
from core.grid import ExperimentGrid
from core.configs import AlgoConfig

gridsearch_name = "stage5_sweep_dicecewt_nbatch"
# I had written the function get_info() in grid.py to get the grid parameters
# in a more table friendly format, but it depended on the user specifying table friendly
#  keys within the experiment.yaml definition. Instead, lets just define a mapping for each key and type of value
#   in one spot as a function or class. It can be a growing function as I try to test more hyperparameters in the future
run_params = grid.run_params(*grid.get_info())

ds = Dataset(dataset_name)
runs_to_check = ["run1"]
runs = []
for run_info in run_params:
    run_loc = f"{grid.experiment_name}/{run_info['run_name']}"
    print(f"STARTING {run_loc}")
    exp = Experiment.from_run_dir(run_loc, ds)
    exp.predict()
    exp.cases #builds the case list for the first time and caches for quick access
    runs.append({**run_info, "exp": exp})


def compile_metrics(runs):
    compiled_metrics = defaultdict(list)
    for run in tqdm(runs, total=len(runs)):
        compiled_metrics["run"].append(run["run_name"])
        # creates columns for each grid parameter that was varied 
        for hyperparam_key, hyperparam_val in run["training"].items():
            compiled_metrics[hyperparam_key].append(hyperparam_val)
        fold_data = analyze_unified_mlruns(run["exp"].run_dir)
        aggregated = aggregate_metrics(fold_data)
        by_prefix = defaultdict(dict)
        for metric_name in aggregated:
            if "/" in metric_name:
                prefix = metric_name.split("/")[0]
                by_prefix[prefix][metric_name] = aggregated[metric_name]
            else:
                by_prefix["other"][metric_name] = aggregated[metric_name]

        for prefix in sorted(by_prefix.keys()):
            for metric_name in sorted(by_prefix[prefix].keys()):
                metric_data = aggregated[metric_name]

                if "stats" not in metric_data:
                    continue

                stats = metric_data["stats"]
                for stat in ["mean", "std", "min", "max"]:
                    compiled_metrics[f"{metric_name}_{stat}"].append(
                        np.round(stats[stat], 4)
                    )

                fold_values = [
                    (fold_num, metric_data[fold_num])
                    for fold_num in sorted(fold_data.keys())
                    if fold_num in metric_data
                ]
                if "val_class" not in metric_name:
                    continue
                for fold_num, values in fold_values:
                    compiled_metrics[f"fold{fold_num}-{metric_name}_final"].append(
                        np.round(values[-1], 4)
                    )
                    compiled_metrics[f"fold{fold_num}-{metric_name}_max"].append(
                        np.round(np.max(values), 4)
                    )

    # Renaming the columns into something more user friendly for when I send this to people
    metrics = pd.DataFrame(compiled_metrics).set_index("run")
    col_remap = {}
    for col in metrics.columns:
        # lesion accs
        new_col = re.sub(r"^(.*)(val_)(class/acc_0)(_.+)", r"\1lesion/acc\4", col)
        col_remap[col] = new_col
        # rim accs
        col_remap[col] = re.sub(
            r"^(.*)(val_)(class/acc_1)(_.+)", r"\1rim/acc\4", new_col
        )

    metrics = metrics.rename(columns=col_remap)
    col_order_cats = defaultdict(list)
    for col in metrics.columns:
        if "acc_min" in col or "loss_max" in col:
            continue
        if "rim" in col:
            col_order_cats["rim"].append(col)
        elif "lesion" in col:
            col_order_cats["lesion"].append(col)
        elif "train" in col:
            col_order_cats["loss"].append(col)
        elif "val" in col:
            col_order_cats["val"].append(col)
        else:
            col_order_cats["run_info"].append(col)

    column_order = []
    for cat in ["run_info", "rim", "lesion", "loss", "val"]:
        column_order.extend(col_order_cats[cat])
    metrics = metrics[column_order]

    return metrics
```

An example of files I produced from previous versions of the above:
- /home/srs-9/Projects/prl_project/analysis/stage3_numcrops_bkd_constwt115_metrics.csv
- /home/srs-9/Projects/prl_project/analysis/stage2_numcrops_dicece_metrics.csv
- /home/srs-9/Projects/prl_project/notebooks/gridsearch_metrics_w_dice.csv


GOing forward I think all stages can be compiled into one list of dictionaries (values for key "run_name" which were originally just like "run1" will have to be turned into things like "stage4_sweep_dicece_wts/run1").

```python
train_root = Path(os.environ["PRL_TRAIN_ROOT"])

def load_grid(ds_name, grid_name):
    config = train_root / ds_name / grid_name / "experiment_config.yaml"
    return ExperimentGrid.from_config(config)

gridsearch_experiments = [
    "stage4_sweep_dicece_wts",
    "stage5_sweep_dicecewt_nbatch",
    "stage3_numcrops_bkd_constwt115",
    "stage1_crop_lr_sweep",
    "stage2_numcrops_dicece",
]

dataset_name = "roi_train2"
for gridsearch_name in gridsearch_experiments:
    # gridsearch_name = "stage5_sweep_dicecewt_nbatch"
    grid = load_grid(dataset_name, gridsearch_name)
    run_params = grid.run_params(*grid.get_info())
    ds = Dataset(dataset_name)
    runs_to_check = ["run1"]
    runs = []
    for run_info in run_params:
        run_loc = f"{grid.experiment_name}/{run_info['run_name']}"
        print(f"STARTING {run_loc}")
        exp = Experiment.from_run_dir(run_loc, ds)
        exp.predict()
        exp.cases #builds the case list for the first time and caches for quick access
        runs.append({**run_info, "exp": exp})

# clear_output()

# Caller should be able to decide which parameters should become columns in the final output
params_to_gather = [
    "learning_rate",
    "crop_ratios",
    "num_crops_per_image",
    "batch_size",
    "loss#weight",
    "loss#include_background",
]

#* Examples:
# this should return a str like "$torch.tensor([1.0, 3.0, 5.0]).cuda()"
wt_str = runs[0]["exp"].hyper_params["loss#weight"] 
# The tensor's string can be made to be nicer for analysis tables with:
str_tensor = re.match(r"\$torch\.tensor\(\[(.+)\]\).+", wt_str)[1]
weights = [float(w) for w in str_tensor.split(",")]
# ^^^ so if there were a class of some sort that kept track of things like this for display, that'd be helpful. Maybe itd make sense inside configs.py since it's so closely tied?
```

I think the compute_metrics function I wrote here should become part of the same suite of things in /home/srs-9/Projects/prl_project/src/scripts/compute_performance_metrics.py. One of the csv files I produced from compute_performance_metrics.py is: /home/srs-9/Projects/prl_project/scratch/roi_train2_stage1croplrsweep_run5_performance_metrics.csv. 

Something like that could be produced except instead of being along cases it could be along runs just the other code which aggregates mlflow runs

My goal is to make it less overwhelming to approach analyzing all these runs. Since these are a ton of asks adding up to a complex job, lets plan it step by step and first think about how to digest the data.