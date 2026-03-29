# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRL (Perilesional Rim Lesion) detection pipeline for 3D medical image segmentation using MONAI Auto3DSeg with SegResNet. Detects paramagnetic rim lesions in MS patients from FLAIR and phase MRI. Two label classes: lesion (1) and rim (2). Rim is extremely rare (~0.5-1% of foreground voxels).

## Environment & Setup

```bash
source setup_env.sh  # Sets PRL_PROJECT_ROOT, PRL_DATA_ROOT, PRL_TRAIN_ROOT
pip install -e .     # Installs `prl` CLI entry point
```

Python environment: `~/.virtualenvs/monai/bin/python`

Key dependencies: MONAI (Auto3DSeg), nibabel, pandas, MLflow, numpy, PyYAML, attrs, click, loguru. External tools: FSL (fslroi, fslmaths, fslstats), C3D.

## Architecture

### Core Classes (`src/core/`)

- **`Dataset`** (`dataset.py`) — Represents a named dataset (e.g., "roi_train2"). Owns subject lists, fold assignments, `create_datalist()`, and a `cases` DataFrame with fully resolved paths. Accepts optional `preprocess` parameter for path resolution (defaults to dataset.yaml defaults). Has `Subject` helper class for per-subject data access.
- **`Experiment`** (`experiment.py`) — A single training run. Owns ROI creation (`create_rois`), data preparation (`prepare_data`), setup, training, and prediction. `cases` property delegates to `Dataset.cases` and adds an `inference` column with model prediction paths.
- **`ExperimentGrid`** (`grid.py`) — HPO management. Generates Cartesian product of parameters, creates run directories, launches locally or on HPC. Routes preprocessing through temporary Experiment instances.
- **`PreprocessingConfig`** / **`AlgoConfig`** (`configs.py`) — Attrs dataclasses for pipeline configuration. `PreprocessingConfig` is frozen/hashable. `AlgoConfig` covers the full `hyper_parameters.yaml` with subclasses `SegResNetConfig`, `SwinUNETRConfig`, `DiNTSConfig` registered in `_ALGO_REGISTRY`. Use `attrs.evolve()` to create variants, `AlgoConfig.from_dict()` for deserialization, `AlgoConfig.from_template()` for algorithm defaults.

### Configuration System

**Single source of truth:** Each dataset has one `dataset.yaml` in `training/{name}/`. All paths, fold params, and defaults are defined there. No more `train_home` duplication across configs.

**Path derivation:**
- `source_home` = `PROJECT_ROOT / "training" / name` (templates, dataset.yaml)
- `work_home` = `TRAIN_ROOT / name` (run directories with model outputs)
- `data_root` = `DATA_ROOT` (subject imaging data)

**Token expansion:** JSON/JSONC/YAML configs use `${PROJECT_ROOT}`, `${DATA_ROOT}`, `${TRAIN_ROOT}` tokens expanded at load time by `load_config()`. Always use `load_config()` instead of raw `json.load()`.

Three roots (set via env vars with defaults):
- `PRL_PROJECT_ROOT` → source code, configs (`/home/srs-9/Projects/prl_project`)
- `PRL_DATA_ROOT` → subject imaging data (`/media/smbshare/srs-9/prl_project/data`)
- `PRL_TRAIN_ROOT` → training outputs (`/media/smbshare/srs-9/prl_project/training`)

## CLI (`prl`)

```bash
prl preprocess roi_train2 [--expand-xy 20] [--expand-z 2] [--images flair --images phase] [--processes 12]
prl train roi_train2 [--run-dir PATH] [--epochs 500] [--lr 0.0002] [--images flair --images phase]
prl grid roi_train2 experiment.yaml [--dry-run] [--launch] [--hpc]
prl predict /path/to/run_dir [--fold N]
prl metrics /path/to/run_dir [--test-only] [--output-csv PATH] [--print]
```

## Pipeline Stages

1. **Preprocess** — `prl preprocess roi_train2`
   - `create_rois`: Crops lesion ROIs with FSL using expand_xy/expand_z
   - `create_datalist`: Stratified fold split (creates datalist_template.json once)
   - `prepare_data`: Stacks channels, produces datalist_xy{X}_z{Z}.json
2. **Train** — `prl train roi_train2` or `python training/roi_train2/train.py --run-dir PATH`
   - MONAI AutoRunner with SegResNet, 5-fold CV
3. **Predict** — `prl predict /path/to/run_dir`
4. **Evaluate** — `prl metrics /path/to/run_dir`

## CLI Design Convention

Scripts take a single positional `run_dir` (or dataset name) argument and derive everything else from configs. Do not add separate `--datalist`, `--dataroot` flags.

## Key Helpers

- `src/helpers/paths.py` — `load_config()`, centralized path constants
- `src/helpers/shell_interface.py` — `command()`, `run_if_missing()` for shell execution with dry-run support
- `src/helpers/parallel.py` — `BetterPool` for graceful multiprocessing
- `my_python_utils` — User's personal utility package (located at ~/python/my_python_utils.py; it's on my PYTHONPATH)

## Data Layout

Subject folders: `$DATA_ROOT/sub{id}-{session}/` containing NIfTI images and per-lesion subfolders with cropped ROIs. The datalist template stores image-agnostic directory paths and explicit `case_type` ("PRL" or "Lesion"); the image stack prefix is determined by `PreprocessingConfig.images` at prepare-data time.

## Training Run Structure

```
$TRAIN_ROOT/roi_train2/run2/
├── label_config.json, monai_config.json    # Auto-generated by Experiment.setup()
├── datalist_flair.phase_xy20_z2.json       # Named with image prefix + expansion params
├── datastats_by_case.yaml                  # Per-case MONAI statistics
├── mlruns/                                 # MLflow tracking
├── segresnet_0/ through segresnet_4/       # Per-fold model outputs
├── fold_predictions/fold{N}/               # Validation inference
├── ensemble_output/                        # Test set inference
└── performance_metrics.csv                 # Per-case metrics
```

## Directory Layout

```
src/
├── cli.py                    # Click CLI entry point
├── core/
│   ├── configs.py            # PreprocessingConfig, AlgoConfig, SegResNetConfig, etc. (attrs)
│   ├── dataset.py            # Dataset class (owns cases DataFrame + Subject)
│   ├── experiment.py         # Experiment class (training runs, inference paths)
│   └── grid.py               # ExperimentGrid class
├── analysis/
│   ├── __init__.py           # Public API
│   ├── loaders.py            # load_run_data() — build experiment data dicts
│   ├── _cache.py             # Orthogonal caching layer for run data
│   ├── compile.py            # compile_all_metrics, compile_experiment_metrics, compile_grid_metrics
│   ├── metrics/
│   │   ├── __init__.py       # Re-exports from submodules
│   │   ├── performance.py    # confusion matrix, derived metrics, casewise stats, performance_metrics()
│   │   ├── mlflow.py         # MLflow metric loading, aggregation, mlflow_metrics()
│   │   └── display.py        # format_param_value, rename_metric, order_columns
│   └── image/
│       ├── __init__.py       # Re-exports from submodules
│       ├── geometry.py       # convex hull, enclosing sphere radius
│       ├── lesion_analysis.py  # rim extraction, PRL case analysis, bounding box parsing
│       └── plotting.py       # 3D visualization (plot_lesion_rim_3d)
├── helpers/                  # paths.py, shell_interface.py, parallel.py, utils.py
├── preprocessing/            # create_rois.py, create_datalist.py, prepare_training_data.py
└── scripts/                  # Legacy shims (re-export from analysis.*), inference, fold predictions

training/roi_train2/
├── dataset.yaml              # Single source of truth for dataset config
├── datalist_template.json    # Fold assignments (created once, never modified)
├── algorithm_templates/      # MONAI Auto3DSeg templates
└── train.py                  # Thin wrapper → Experiment.train()
```

## Data Analysis

analysis package

## HPO

`ExperimentGrid` in `src/core/grid.py` handles HPO. Define an experiment YAML:

```yaml
dataset: roi_train2
experiment_name: my_sweep
param_grid:
  training:
    learning_rate: [0.0001, 0.0002]
    crop_ratios: [null, [1, 1, 4]]
  # preprocessing params (expand_xy, expand_z, images) can also be swept:
  # preprocessing:
  #   expand_xy: [10, 20]
  #   images: [[flair, phase], [flair, phase, t1]]
```

Then: `prl grid roi_train2 experiment.yaml --launch`

