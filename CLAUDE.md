# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRL (Perilesional Rim Lesion) detection pipeline for 3D medical image segmentation using MONAI Auto3DSeg with SegResNet. Detects paramagnetic rim lesions in MS patients from FLAIR and phase MRI. Two label classes: lesion (1) and rim (2). Rim is extremely rare (~0.5-1% of foreground voxels).

## Environment & Setup

```bash
source setup_env.sh  # Sets PRL_PROJECT_ROOT, PRL_DATA_ROOT, PRL_TRAIN_ROOT
```

Python environment: `~/.virtualenvs/monai/bin/python`

Key dependencies: MONAI (Auto3DSeg), nibabel, pandas, MLflow, numpy, PyYAML. External tools: FSL (fslroi, fslmaths, fslstats), C3D.

## Path Configuration System

All paths flow through `helpers/paths.py`. JSON configs use `${PROJECT_ROOT}`, `${DATA_ROOT}`, `${TRAIN_ROOT}` tokens that are expanded at load time by `load_config()`. Always use `load_config()` instead of raw `json.load()`.

Three roots (set via env vars with defaults):
- `PRL_PROJECT_ROOT` → source code, configs (`/home/srs-9/Projects/prl_project`)
- `PRL_DATA_ROOT` → subject imaging data (`/media/smbshare/srs-9/prl_project/data`)
- `PRL_TRAIN_ROOT` → training outputs (`/media/smbshare/srs-9/prl_project/training`)

## Pipeline Stages

Each training experiment has two config files in its training directory: `label_config.json` (ROI parameters) and `monai_config.json` (training hyperparameters).

1. **Copy raw data** — `preprocessing/copy_files.py`
2. **Create ROIs** — `preprocessing/create_rois.py label_config.json --processes N`
   Crops lesion bounding boxes with configurable expansion (expand_xy, expand_z)
3. **Create datalist** — `preprocessing/create_datalist.py label_config.json monai_config.json`
   Stratified split of PRL vs lesion-only cases across 5 folds + test set
4. **Prepare training data** — `preprocessing/prepare_training_data.py label_config.json`
   Stacks multi-channel images, outputs `datalist_xy{expand_xy}_z{expand_z}.json`
5. **Train** — `training/roi_train2/train.py [--run-dir RUN_DIR]`
   MONAI AutoRunner with SegResNet, 5-fold CV. Auto-increments run directories.
6. **Generate predictions** — `scripts/generate_fold_predictions.py run_dir`
7. **Compute metrics** — `scripts/compute_performance_metrics.py run_dir`

## CLI Design Convention

Scripts take a single positional `run_dir` (or config path) argument and derive everything else from configs inside that directory. Do not add separate `--datalist`, `--dataroot` flags.

## Key Helpers

- `helpers/paths.py` — `load_config()`, centralized path constants
- `helpers/shell_interface.py` — `command()`, `run_if_missing()` for shell execution with dry-run support
- `helpers/parallel.py` — `BetterPool` for graceful multiprocessing
- `my_python_utils` — User's personal utility package (located at ~/python/my_python_utils.py; it's on my PYTHONPATH)

## Data Layout

Subject folders: `$DATA_ROOT/sub{id}-{session}/` containing NIfTI images and per-lesion subfolders with cropped ROIs. Datalist entries use `"image"` and `"label"` keys; case type (PRL vs lesion-only) is determined by label filename pattern.

## Training Run Structure

```
$TRAIN_ROOT/roi_train2/run2/
├── label_config.json, monai_config.json    # Copied configs
├── datalist_xy20_z2.json                   # Named with expansion params
├── datastats_by_case.yaml                  # Per-case MONAI statistics
├── mlruns/                                 # MLflow tracking
├── segresnet_0/ through segresnet_4/       # Per-fold model outputs
├── fold_predictions/fold{N}/               # Validation inference
├── ensemble_output/                        # Test set inference
└── performance_metrics.csv                 # Per-case metrics
```

## HPO Infrastructure

`scripts/generate_experiments.py` reads an `experiment_config.json` with `param_grid`, generates run directories. `scripts/launch_experiments.py` launches locally or via IBM LSF job arrays (HPC scaffolding in `hpc/`).
