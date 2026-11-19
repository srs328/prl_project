# %%
import matplotlib.pyplot as plt

import os
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import json
import pickle as pkl
from collections import defaultdict

import scripts.compile_run_metrics
import helpers.utils
import scripts.inference
from IPython.display import clear_output

from reload_recursive import reload_recursive
from tqdm.notebook import tqdm

from helpers.parallel import BetterPool

# %%
reload_recursive(scripts.compile_run_metrics)
#TODO Thi
from scripts.compile_run_metrics import EXPERIMENT_KEYS

reload_recursive(scripts.inference)
from scripts.inference import uncrop_predictions

import classification
reload_recursive(classification)
from classification import lesion_analysis as lesion_dx

reload_recursive(helpers.utils)
from helpers.utils import dice_score, get_prl_indices

import helpers.shell_interface
reload_recursive(helpers.shell_interface)
from helpers.shell_interface import open_itksnap_workspace_cmd

import core.dataset
reload_recursive(core.dataset)
from core.dataset import Dataset

import core.experiment
reload_recursive(core.experiment)
from core.experiment import Experiment
from loguru import logger

import scripts
reload_recursive(scripts)
from scripts.compile_run_metrics import load_or_cache_run

# %%
def get_infer_path(dataset, test_case, experiment_name) -> Path:
    case_dir = dataset.lesion_dir(test_case)
    matches = list(case_dir.glob(f"*{experiment_name.replace('/','_')}.nii.gz"))
    if len(matches) > 1:
        logger.warning(f"Found more than 1 case: {','.join(matches)}, returning the first")
    return matches[0]
    
def find_inference(search_path, experiment_name) -> Path:
    matches = list(search_path.glob(f"*{experiment_name.replace('/','_')}.nii.gz"))
    if len(matches) > 1:
        logger.warning(f"Found more than 1 case: {','.join(matches)}, returning the first")
    return matches[0]



# %%
dataset_name = "roi_train2"
dataset = Dataset(dataset_name)

experiment_key, run_name = "stage6", "run1"
experiment_name = f"{EXPERIMENT_KEYS[experiment_key]}/{run_name}"
# experiment_name = "sweep_dicecewts/run1"
# experiment_name = "test_dicece_lambda/run1"
experiment = Experiment.from_run_dir(experiment_name, dataset)

expand_xy: int = experiment.preprocess_config.expand_xy
expand_z: int = experiment.preprocess_config.expand_z
images: tuple[str, ...] = experiment.preprocess_config.images

inference_dataset = Dataset("inference_dataset")

inference_dataset.cases['inference'] = None
for i, case_i in tqdm(inference_dataset.cases.iterrows()):
    inference_dataset.cases.loc[i, "inference"] = find_inference(Path(case_i.subject_dir)/str(case_i.name[1]), experiment_name)

# %%
all_results = {}

for subid in tqdm(inference_dataset.cases.index.get_level_values("subid").unique()):
    # screen for iron returns dict of {lesion_index: bool}
    res = lesion_dx.screen_for_iron(inference_dataset, subid, "inference")
    # Store with the full (subid, lesion_index) tuple key
    for l_idx, val in res.items():
        all_results[(subid, l_idx)] = val

# Map the dictionary to the index to create the column all at once
inference_dataset.cases['has_iron'] = inference_dataset.cases.index.map(all_results).fillna(False)
screened_cases = inference_dataset.cases[inference_dataset.cases['has_iron']]

# %%
full_data = defaultdict(list)
def analyze_wrapper(args):
    dataset, subid, lesion_indices = args
    return lesion_dx.analyze_subject_prl(subid, dataset, "inference", lesion_indices=lesion_indices)

tasks = []
for subid, g in inference_dataset.cases.reset_index().groupby("subid"):
    tasks.append((inference_dataset, subid, g['lesion_index'].to_list()))
with BetterPool(8) as pool:
    results_iterator = pool.imap_unordered(analyze_wrapper, tasks)
    for result in tqdm(results_iterator, total=len(tasks)):
        full_data['prl_stats'].extend(result[0])
        full_data['prl_data'].extend(result[1])

analysis_dir = Path("/home/srs-9/Projects/prl_project/analysis/")
feature_df = pd.DataFrame(full_data['prl_stats'])

savepath = analysis_dir/ f"prl_image_stats-{experiment.id.replace('/', '_')}.csv"
if not savepath.exists():
    feature_df.to_csv(
        savepath,
        index=False
    )

# %%
df = inference_dataset.cases.reset_index().merge(
    feature_df,
    on=["subid", "lesion_index"],
    how="left",
    validate="one_to_one",
    suffixes=("", "_y")
)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# %%
TARGET = "case_type"        # "PRL" or "Lesion"
POS_LABEL = "PRL"

TEST_SIZE = 0.25            # fraction held out
RANDOM_STATE = 42
CV_FOLDS = 5

# FEATURES = [col for col in all_features.columns if 'shape' in col]
FEATURES = [
    "rim_volume_infer",
    "rim_hull_volume_infer",
    "rim_sphere_radius_infer",
    "pca_size",
    "pca_sphericity",

    "pca_planarity",
    "pca_linearity",
    "mean_radius",
    "std_radius",
    "min_radius",
    "max_radius",
    "radial_cv",
]

# %%
TEST_SIZE = 0.1            # fraction held out

model_df = df[df.has_iron_infer].copy()
print(f"Loaded {len(model_df)} rows | {model_df[TARGET].value_counts().to_dict()}")

X = model_df[['subid', 'lesion_index']+FEATURES].values
y = (model_df[TARGET] == POS_LABEL).astype(int).values   # 1 = PRL, 0 = Lesion

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {len(y_train)} ({y_train.sum()} PRL) | Test: {len(y_test)} ({y_test.sum()} PRL)")

index_train = X_train[:, :2]
index_test = X_test[:, :2]
X_train = X_train[:, 2:]
X_test = X_test[:, 2:]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),   # hull/sphere NaN when <4 rim voxels
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",    # important: 48 PRL vs ~117 Lesion
        max_iter=1000,
        random_state=RANDOM_STATE,
    )),
])

model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Test set ---")
print(classification_report(y_test, y_pred, target_names=["Lesion", "PRL"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Lesion", "PRL"]).plot(ax=axes[0])
axes[0].set_title("Confusion matrix (test set)")
RocCurveDisplay.from_predictions(y_test, y_prob, pos_label=1, ax=axes[1])
axes[1].set_title("ROC curve (test set)")
plt.tight_layout()
plt.show()
