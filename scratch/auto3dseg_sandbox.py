"""Sandbox for experimenting with Auto3DSeg parameter passing.

Run this, let a fold start, Ctrl+C, then inspect the fold directory to see
what actually made it into hyper_parameters.yaml and training.log.

Usage:
    python scratch/auto3dseg_sandbox.py

Change WORK_DIR and the input_cfg / train_param / input_extras dicts below,
then rerun to try a new combination. Each run in a fresh directory starts clean.
"""

import os
from pathlib import Path
from monai.apps.auto3dseg import (
    AutoRunner,
    DataAnalyzer,
    BundleGen,
    import_bundle_algo_history,
    export_bundle_algo_history,
)
from monai.auto3dseg import algo_to_pickle
from monai.bundle.config_parser import ConfigParser
from monai.utils.enums import AlgoKeys

# ── Config ─────────────────────────────────────────────────────────────────────

DATALIST  = "/media/smbshare/srs-9/prl_project/training/roi_train2/datalist_flair.phase_xy20_z2.json"
DATAROOT  = "/media/smbshare/srs-9/prl_project/data"
WORK_DIR  = "/tmp/auto3dseg_sandbox"   # ← change this between experiments

# Params that go into the input dict → fill_template_config → hyper_parameters.yaml
# These survive regardless of type (scalars, lists, dicts all work here)
input_extras = {
    "roi_size": [44, 44, 8],
    "learning_rate": 0.0002,
    "num_epochs": 10,           # small for quick inspection
    "num_warmup_epochs": 1,
    "num_epochs_per_validation": 1,
    "auto_scale_batch": False,
    "batch_size": 2,            # → swapped to num_crops_per_image by segmenter
    # "crop_ratios": [1, 1, 4],
    # "loss": {
    #     "_target_": "DiceCELoss",
    #     "include_background": False,
    #     "weight": "$torch.tensor([1.0, 1.0, 5.0]).cuda()",
    #     "squared_pred": True,
    #     "smooth_nr": 0,
    #     "smooth_dr": 1e-05,
    #     "softmax": True,
    #     "sigmoid": False,
    #     "to_onehot_y": True,
    # },
}

# Params that go through set_training_params → CLI overrides (scalars only)
# For this project, leave empty — everything goes through input_extras above
train_param = {}

# ── Setup ───────────────────────────────────────────────────────────────────────

os.makedirs(WORK_DIR, exist_ok=True)

input_cfg = {
    "modality": "MRI",
    "datalist": DATALIST,
    "dataroot": DATAROOT,
    **input_extras,
}

input_yaml = os.path.join(WORK_DIR, "input.yaml")
ConfigParser.export_config_file(input_cfg, input_yaml)
print(f"Wrote input.yaml to {input_yaml}")
print("input_cfg:", input_cfg)

# ── Option A: Use AutoRunner (simplest, mimics your train.py) ──────────────────

runner = AutoRunner(
    work_dir=WORK_DIR,
    algos=["segresnet"],
    input=input_cfg,
)
if train_param:
    runner.set_training_params(train_param)
runner.run()

# ── Option B: Step-by-step (use this if you want finer control) ───────────────
# Uncomment to use instead of Option A.

# datastats_file = os.path.join(WORK_DIR, "datastats.yaml")
# if not os.path.exists(datastats_file):
#     analyser = DataAnalyzer(DATALIST, DATAROOT, output_path=datastats_file)
#     analyser.get_all_case_stats()

# bundle_gen = BundleGen(
#     algo_path=WORK_DIR,
#     data_stats_filename=datastats_file,
#     data_src_cfg_name=input_yaml,
# )
# bundle_gen.generate(WORK_DIR, num_fold=5)
# history = bundle_gen.get_history()
# export_bundle_algo_history(history)

# history = import_bundle_algo_history(WORK_DIR, only_trained=False)
# for algo_dict in history:
#     algo = algo_dict[AlgoKeys.ALGO]
#     algo.train(train_param or {})
#     try:
#         acc = algo.get_score()
#         algo_to_pickle(algo, template_path=algo.template_path, best_metric=acc)
#     except Exception:
#         pass  # fold killed before completion, score unavailable

# ── After running ──────────────────────────────────────────────────────────────
# Inspect results:
#   cat WORK_DIR/input.yaml                            → what was passed in
#   cat WORK_DIR/segresnet_0/configs/hyper_parameters.yaml  → post-fill (post-swap after training)
#   head -150 WORK_DIR/segresnet_0/model/training.log  → pre-swap values + swap log lines
