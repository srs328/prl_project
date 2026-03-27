# Grid Search Plan for PRL Rim Detection

## Context

Run3 (500 epochs, lr=0.0002) improved rim F1 from 0.577→0.591 over Run2 (250 epochs) — a modest gain from more training alone. The rim class is extremely rare (~0.5-1% of foreground voxels), and the current pipeline uses equal crop sampling across all 3 classes, meaning most crops contain zero rim voxels. A targeted grid search over **crop oversampling** and **learning rate** should yield significant improvements by directly addressing the class imbalance problem.

## Run2 vs Run3 Summary

| Metric (Test, PRL cases) | Run2 | Run3 | Delta |
|---------------------------|------|------|-------|
| Sensitivity | 0.584 | 0.601 | +2.9% |
| Precision | 0.613 | 0.633 | +3.1% |
| F1 Score | 0.577 | 0.591 | +2.4% |
| Specificity | 0.876 | 0.888 | +1.3% |

Only change: epochs 250→500. All other params identical.

## Parameter Selection

### 1. `crop_ratios` — Rim class oversampling (HIGHEST IMPACT)

Currently `null` → equal probability across background/lesion/rim. Since `RandCropByLabelClassesd` centers crops on class voxels according to `ratios`, setting `[1, 1, 4]` makes rim-centered crops 4x more likely. This is the single highest-impact lever for rare-class detection.

- Config key: `train_param.crop_ratios` (flows through `input_config` → `config.update()` → `hyper_parameters.yaml`)
- Values: `null` (baseline), `[1, 1, 4]`, `[1, 1, 8]`

**Bug fix needed**: current `monai_config.jsonc` has key `crop_ratio` (singular) — must be `crop_ratios` (plural) to match what segmenter.py reads at line 338/733/888.

### 2. `learning_rate` — Optimizer step size

Standard sweep parameter. With crop_ratios changing the training distribution significantly, the optimal LR likely shifts. Current 0.0002 may be suboptimal.

- Config key: `train_param.learning_rate`
- Values: `0.0001`, `0.0002` (baseline), `0.0005`

### Grid: 3 × 3 = 9 runs

| Run | crop_ratios | learning_rate |
|-----|-------------|---------------|
| 1 | null | 0.0001 |
| 2 | null | 0.0002 |
| 3 | null | 0.0005 |
| 4 | [1,1,4] | 0.0001 |
| 5 | [1,1,4] | 0.0002 |
| 6 | [1,1,4] | 0.0005 |
| 7 | [1,1,8] | 0.0001 |
| 8 | [1,1,8] | 0.0002 |
| 9 | [1,1,8] | 0.0005 |

Run 2 is essentially a repeat of Run3 (for baseline comparison within the grid).

**Compute estimate**: 9 runs × 5 folds × ~2 hrs = ~90 GPU-hours. On HPC with 9+ V100 GPUs, wall time ~10 hours.

### Future: Loss class weights (noted for later)

`DiceCELoss(weight=[1.0, 1.0, 5.0])` would weight rim class higher in the loss function. Requires modifying `train.py` to run AutoRunner in two phases (generate configs → patch `hyper_parameters.yaml` → train), since the `loss` dict can't flow through `train_param` → CLI args cleanly. Worth trying after this grid search.

## AutoRunner Built-in HPO Assessment

AutoRunner has built-in NNI (Neural Network Intelligence) HPO support via `hpo=True` + `set_nni_search_space()`. After analysis, **we should NOT use it** for this grid search:

1. **Same list-mangling problem**: NNI HPO goes through the same `BundleAlgo.train()` → `check_and_set_optional_args()` → CLI args path. `crop_ratios=[1,1,4]` would still get mangled by Python Fire.
2. **Overkill**: NNI adds a dependency, requires an NNI config, and manages its own trial infrastructure. Our 9-run grid is simple enough that separate run directories + LSF job arrays are cleaner.
3. **Less control**: NNI manages trial lifecycle internally; with separate run dirs, each run is independently restartable, inspectable, and debuggable.

**Conclusion**: Our current approach (generate_experiments.py + launch_experiments.py + LSF job arrays) is the right architecture for this use case.

## Critical Architecture Notes (from debugging)

### `input` dict vs `set_training_params()` — Two different paths

This is the most important thing to understand:

- **`input` dict** (passed to `AutoRunner(input={...})`) → flows to `algo.fill_template_config()` → written into `hyper_parameters.yaml` via `config.update(input_config)`. **This is where list-valued params like `crop_ratios` and `roi_size` must go.**
- **`set_training_params()`** → flows to `algo.train(train_params=...)` → converted to CLI args → override `hyper_parameters.yaml` at runtime. Scalars only — lists get mangled by Fire, dicts are rejected.

The fix in `train.py` (already applied) moves list-valued params from `train_param` to `input_dict` automatically.

### AutoRunner caching (`cache.yaml`)

AutoRunner caches phase completion in `work_dir/cache.yaml`. If a run directory already exists with `algo_gen: true`, it skips config generation even if params changed. **For grid search runs, each run gets a fresh directory so this isn't an issue.** But for manual reruns of a single experiment, delete `cache.yaml` or the entire run directory.

## Implementation Steps

### Step 1: Fix `crop_ratio` → `crop_ratios` in monai_config.jsonc

**File**: `training/roi_train2/monai_config.jsonc`

Change `"crop_ratio": null` to `"crop_ratios": null`.

### Step 2: Handle `null` values in `generate_experiments.py`

**File**: `src/scripts/generate_experiments.py`

When grid value is `None`, delete the key from the config so the key is absent from `train_param` entirely. Already handled by `delete_nested_key()`.

### Step 3: Create `experiment_config.json`

**File**: `training/roi_train2/experiment_config.json` (new file)

```json
{
  "experiment_name": "stage1_crop_lr_sweep",
  "base_label_config": "label_config.jsonc",
  "base_monai_config": "monai_config.jsonc",
  "param_grid": {
    "monai": {
      "train_param.crop_ratios": [null, [1, 1, 4], [1, 1, 8]],
      "train_param.learning_rate": [0.0001, 0.0002, 0.0005]
    }
  }
}
```

### Step 4: Clean up failed test run (run5)

Delete `$TRAIN_ROOT/roi_train2/run5/` which has corrupted `hyper_parameters.yaml` from the CLI mangling debugging.

### Step 5: Verify experiment generation

```bash
cd training/roi_train2
PYTHONPATH=$PRL_PROJECT_ROOT/src:$PYTHONPATH python -m scripts.generate_experiments experiment_config.json --dry-run
# Verify 9 runs with correct param combinations
```

Then generate for real:
```bash
PYTHONPATH=$PRL_PROJECT_ROOT/src:$PYTHONPATH python -m scripts.generate_experiments experiment_config.json
```

### Step 6: Verify param flow locally (single run)

Run one experiment locally (e.g., run5 = `crop_ratios=[1,1,4]`, `lr=0.0002`) to confirm:
- `crop_ratios` appears correctly in `hyper_parameters.yaml` as a list `[1, 1, 4]`
- Training starts without errors
- Stop after a few epochs once confirmed

```bash
PYTHONPATH=$PRL_PROJECT_ROOT/src:$PYTHONPATH python -m scripts.launch_experiments experiment_config.json --run-key run5
# Ctrl+C after confirming training starts
```

### Step 7: Launch full grid on HPC

```bash
PYTHONPATH=$PRL_PROJECT_ROOT/src:$PYTHONPATH python -m scripts.launch_experiments experiment_config.json --hpc --dry-run
# Review generated submit_array.sh
PYTHONPATH=$PRL_PROJECT_ROOT/src:$PYTHONPATH python -m scripts.launch_experiments experiment_config.json --hpc
```

### Step 8: Post-experiment analysis

For each completed run:
```bash
python src/scripts/generate_fold_predictions.py <run_dir>
python src/scripts/compute_performance_metrics.py <run_dir>
```

Compare `performance_metrics.csv` across all 9 runs, focusing on rim F1 and sensitivity.

## Files to Modify

| File | Change |
|------|--------|
| `training/roi_train2/monai_config.jsonc` | Fix `crop_ratio` → `crop_ratios` |
| `training/roi_train2/train.py` | List-valued params moved to `input_dict` |
| `src/scripts/generate_experiments.py` | Handles `null` grid values via key deletion |
| `src/scripts/launch_experiments.py` | LSF job array + `--run-key` support |
| `training/roi_train2/experiment_config.json` | New — grid search config |
| `src/helpers/paths.py` | JSONC support |
| `hpc/setup_env_hpc.sh` | UMass cluster env setup |

## Verification

1. `generate_experiments.py --dry-run` shows 9 runs with correct param combos
2. Inspect a generated `monai_config.json` in a run dir — confirm `crop_ratios` is set correctly (as a list, not string)
3. Run one config locally to verify `crop_ratios` flows through to `hyper_parameters.yaml` and training starts
4. After full grid: compare rim F1/sensitivity across all 9 runs

---

# Auto3DSeg SegResNet Infrastructure Guide

## Overview

MONAI Auto3DSeg automates 3D medical image segmentation pipeline setup. For this project, it uses the **SegResNet** algorithm with deep supervision (`SegResNetDS`). The framework handles data analysis, config generation, training, and inference across multiple cross-validation folds.

## Architecture: How the Pieces Fit Together

```
User configs                    MONAI AutoRunner                     Per-fold training
─────────────                   ────────────────                     ─────────────────
monai_config.json ──┐
                    ├──→ AutoRunner ──→ DataAnalyzer ──→ datastats.yaml
label_config.json ──┘       │                                │
                            ├──→ algo.py fill_template_config()
                            │         │
                            │         ├── reads datastats.yaml (image sizes, spacing, etc.)
                            │         ├── reads input_config (your input dict values)
                            │         ├── auto-adjusts network settings (roi, filters, batch)
                            │         └── writes hyper_parameters.yaml (per fold)
                            │
                            └──→ BundleAlgo.train() ──→ segmenter.py
                                      │                     │
                                      │                     ├── reads hyper_parameters.yaml
                                      └── CLI overrides ──→ ├── builds transforms, model, loss
                                          (scalars only)    └── runs training loop
```

## Parameter Flow in Detail

### Stage 1: Your configs → AutoRunner (TWO separate paths)

**`train.py`** loads configs, splits params into two paths:

```python
train_param = monai_config['train_param']
algos = train_param.pop("algos")

# List-valued params must go in input_dict (not train_param)
# because train_param → CLI args, and Fire mangles lists
input_dict = {"modality": "MRI", "datalist": str(datalist_file), "dataroot": str(dataroot)}
for key in list(train_param):
    if isinstance(train_param[key], list):
        input_dict[key] = train_param.pop(key)

runner = AutoRunner(work_dir=..., algos=algos, input=input_dict, ...)
runner.set_training_params(train_param)  # scalars only → CLI overrides
runner.run()
```

**Two separate paths**:
- **`input` dict** → written to internal YAML → read by `algo.fill_template_config()` → baked into `hyper_parameters.yaml`. **Lists (crop_ratios, roi_size) go here.**
- **`set_training_params()`** → passed to `algo.train()` as CLI overrides → applied at runtime on top of `hyper_parameters.yaml`. **Scalars only (learning_rate, num_epochs).**

### Stage 2: algo.py — Config generation

`SegresnetAlgo.fill_template_config()` in `algorithm_templates/segresnet/scripts/algo.py`:

1. **Loads** the combined input YAML as `input_config` (line 61)
2. **Pops** known keys from `input_config` and processes them:
   - `dataroot`, `datalist`, `modality` → basic setup
   - `resample_resolution`, `resample_mode` → spacing config
   - `roi_size`, `batch_size` → override auto-computed values
   - `auto_scale_batch/roi/filters` → GPU-aware auto-adjustment
3. **Auto-adjusts** network settings via `auto_adjust_network_settings()`:
   - Computes `roi_size`, `levels`, `init_filters`, `batch_size` from image size + spacing
   - Your explicit `roi_size`/`batch_size` overrides these if provided
4. **Determines** `blocks_down` from `levels` (adapts network depth to roi_size)
5. **Determines** `crop_mode` — "ratio" if roi_size < 0.8 × image_size, else "rand"
6. **Merges** remaining `input_config` into `config` via `config.update(input_config)` (line 271)
   - **This is the key line**: any input dict key that wasn't explicitly popped earlier gets injected into the final config. This is how `crop_ratios`, `num_epochs`, `learning_rate`, etc. make it through.
7. **Writes** the final config to `segresnet_<fold>/configs/hyper_parameters.yaml`

**Important**: The `config.update(input_config)` is a **shallow merge** on a plain dict. For nested keys like `loss` (which is itself a dict), passing `loss: {weight: [1,1,5]}` would REPLACE the entire loss dict, not merge into it. However, when the dict is written to YAML via ConfigParser (lines 276-295), the `#` notation in key names IS supported as nested key access (e.g., `loss#weight` would set `loss.weight`).

### Stage 3: BundleAlgo.train() — Training launch

After config generation, `BundleAlgo.train(train_params=...)` launches training:

1. Converts `train_params` to CLI arguments via `check_and_set_optional_args()`
   - **Scalars**: converted to `--key=value`
   - **Lists**: converted to Fire-compatible format `--key=[v1,v2,v3]` — **BUT Fire mangles these**
   - **Dicts**: **REJECTED** with `ValueError("Nested dict is not supported")`
2. Runs `python scripts/train.py run --config_file=configs/hyper_parameters.yaml --key1=val1 ...`
3. Python Fire parses CLI args → passed as `config_dict` override to `Segmenter`

**Implication**: Only scalars flow cleanly through this path. Lists and dicts must go through the `input` dict path instead.

### Stage 4: segmenter.py — Training execution

`Segmenter.__init__()` in `algorithm_templates/segresnet/scripts/segmenter.py`:

1. **Loads** `hyper_parameters.yaml` via `parse_input_config()` (line 599)
2. **Applies** CLI overrides from `config_dict`
3. **Sets defaults** for any missing keys (lines 830-940)
4. **Auto-scales** network again if `auto_scale_allowed` + `auto_scale_batch/roi/filters` (lines 642-659)
5. **Builds** model, loss, optimizer, sliding window inferrer

## Key Config Parameters Reference

### hyper_parameters.yaml — Complete parameter map

These are all the parameters in the generated config, grouped by function:

#### Data & I/O
| Key | Default | Description |
|-----|---------|-------------|
| `input_channels` | from data | Number of input image channels (2 for FLAIR+phase) |
| `output_classes` | from data | Number of output classes (3: bg, lesion, rim) |
| `data_file_base_dir` | from config | Root path for image data |
| `data_list_file_path` | from config | Path to datalist JSON |
| `modality` | "mri" | Imaging modality |
| `fold` | 0-4 | Current CV fold |

#### Preprocessing & Augmentation
| Key | Default | Description |
|-----|---------|-------------|
| `resample` | auto | Whether to resample to common spacing |
| `resample_resolution` | from stats | Target voxel spacing [x, y, z] |
| `normalize_mode` | "meanstd" | Intensity normalization ("meanstd" for MRI, "range" for CT) |
| `intensity_bounds` | from stats | Clipping bounds for normalization |
| `orientation_ras` | false | Reorient to RAS |
| `crop_foreground` | true | Crop to foreground before patching |

#### Cropping (Critical for rare-class detection)
| Key | Default | Description |
|-----|---------|-------------|
| `crop_mode` | "ratio" or "rand" | "ratio" = `RandCropByLabelClassesd` (class-balanced), "rand" = `RandSpatialCropd` |
| `crop_ratios` | null | Per-class sampling weights for `RandCropByLabelClassesd`. null = equal across classes. `[1,1,4]` = 4x rim oversampling |
| `roi_size` | auto-computed | Patch size for random cropping [x, y, z] |
| `num_crops_per_image` | 1 | Number of crops extracted per image per step |
| `cache_class_indices` | auto | Pre-cache voxel indices per class for faster cropping |
| `max_samples_per_class` | 10 × num_epochs | Max cached voxel indices per class |
| `crop_add_background` | true | Include background as a crop class |

**How `crop_mode="ratio"` works**: `RandCropByLabelClassesd` picks a random class according to `ratios` (or uniform if null), then centers the crop on a random voxel of that class. With 3 output classes and null ratios, each class has ~33% chance. With `[1, 1, 4]`, rim gets `4/6 ≈ 67%` chance of being the center class.

#### Network Architecture
| Key | Default | Description |
|-----|---------|-------------|
| `network._target_` | SegResNetDS | Network class |
| `network.init_filters` | 32 | Base filter count (doubles each level) |
| `network.blocks_down` | [1, 2, 4] | Residual blocks per encoder level (auto-set from roi_size) |
| `network.norm` | INSTANCE_NVFUSER | Normalization type |
| `network.dsdepth` | 4 | Deep supervision depth |

#### Training Loop
| Key | Default | Description |
|-----|---------|-------------|
| `learning_rate` | 0.0002 | Initial LR for AdamW |
| `batch_size` | 1 | Images per batch (see auto_scale interaction below) |
| `num_epochs` | 500 | Total training epochs (may be auto-adjusted) |
| `num_warmup_epochs` | 1 | Cosine warmup epochs |
| `num_epochs_per_validation` | 1 | Validate every N epochs |
| `num_epochs_per_saving` | 1 | Save checkpoint every N epochs |
| `early_stopping_fraction` | 0.001 | Stop if val metric improves less than this |
| `amp` | true | Mixed precision training |
| `channels_last` | true | Memory format optimization |

#### Loss Function
| Key | Default | Description |
|-----|---------|-------------|
| `loss._target_` | DiceCELoss | Loss class |
| `loss.include_background` | true | Include background in Dice computation |
| `loss.squared_pred` | true | Square predictions in Dice denominator (more stable) |
| `loss.smooth_nr` | 0 | Numerator smoothing |
| `loss.smooth_dr` | 1e-5 | Denominator smoothing |
| `loss.softmax` | true | Apply softmax (mutually exclusive with sigmoid) |
| `loss.sigmoid` | false | Apply sigmoid (for multi-label) |
| `loss.to_onehot_y` | true | Convert labels to one-hot |
| `loss.weight` | none | **Per-class weights** — not set by default. Would be `[1.0, 1.0, 5.0]` to emphasize rim |

#### Optimizer
| Key | Default | Description |
|-----|---------|-------------|
| `optimizer._target_` | torch.optim.AdamW | Optimizer class |
| `optimizer.lr` | @learning_rate | References the learning_rate key |
| `optimizer.weight_decay` | 1e-5 | L2 regularization |

#### Auto-scaling (GPU-aware)
| Key | Default | Description |
|-----|---------|-------------|
| `auto_scale_allowed` | true | Master switch for auto-scaling |
| `auto_scale_batch` | true | Auto-adjust batch size for GPU memory |
| `auto_scale_roi` | false | Auto-adjust roi_size for GPU memory |
| `auto_scale_filters` | false | Auto-adjust init_filters for GPU memory |

## The batch_size / num_crops_per_image Interaction

This is a subtle but important behavior in segmenter.py (lines 1191-1213):

When `auto_scale_allowed=true` and `crop_mode="ratio"`:
```python
# segmenter.py lines 1195-1197
if config["crop_mode"] == "ratio":
    config["num_crops_per_image"] = config["batch_size"]  # e.g., 1
    config["batch_size"] = 1  # always 1 for ratio mode
```

Then epochs are adjusted:
```python
# line 1213
num_epochs = max(1, config["num_epochs"] // min(3, num_crops_per_image))
```

So if you set `num_images_per_batch=2`:
- `batch_size` → 2 initially
- Swap: `num_crops_per_image` = 2, `batch_size` = 1
- `num_epochs` = 500 // min(3, 2) = 250 effective epochs
- Each step: 2 crops from 1 image (more spatial diversity, half the epochs)

With `num_images_per_batch=1` (current):
- `num_crops_per_image` = 1, `batch_size` = 1
- `num_epochs` = 500 // 1 = 500 (unchanged)

## What You Can vs. Can't Control via configs

### Via `input` dict (→ fill_template_config → hyper_parameters.yaml):
- **Lists**: `roi_size`, `crop_ratios`, `resample_resolution`
- **Scalars**: everything below also works here, but scalar-only params are simpler via train_param

### Via `set_training_params()` (→ CLI overrides, scalars only):
- `learning_rate`, `num_epochs`, `num_warmup_epochs`, `num_epochs_per_validation`
- `num_images_per_batch`, `early_stopping_fraction`, `num_workers`
- `auto_scale_batch`, `auto_scale_roi`, `auto_scale_filters` (booleans)

### Cannot set via either path (nested dicts):
- `loss` (dict with `_target_`, params) — rejected by `check_and_set_optional_args()`
- `optimizer` (dict) — same issue
- `network` (dict) — same issue

### Workaround for nested params:
Modify `train.py` to run AutoRunner in phases: generate configs first, patch the YAML, then train. Or use the `#` notation (e.g., `network#init_filters`) which ConfigParser handles as nested keys during YAML writing — but may fail at the Fire CLI parsing stage.

**Note**: `train.py` already auto-routes list-valued params from `train_param` to `input_dict` (the `for key in list(train_param): if isinstance(..., list)` loop).

## Directory Structure (Per Run)

```
$TRAIN_ROOT/roi_train2/run3/
├── label_config.json              # Copied from training home
├── monai_config.json              # Copied from training home
├── datalist_xy20_z2.json          # Copied datalist
├── info.txt                       # Run description
├── datastats.yaml                 # MONAI DataAnalyzer output
├── mlruns/                        # MLflow tracking
│
├── algorithm_templates/           # Template scripts (shared across folds)
│   └── segresnet/
│       ├── configs/
│       │   └── hyper_parameters.yaml   # TEMPLATE (before fill)
│       └── scripts/
│           ├── algo.py                 # Config generation logic
│           ├── segmenter.py            # Training/inference logic
│           ├── train.py                # Entry point (Fire CLI)
│           └── utils.py                # auto_adjust_network_settings()
│
├── segresnet_0/ through segresnet_4/   # Per-fold generated bundles
│   ├── configs/
│   │   └── hyper_parameters.yaml       # GENERATED (filled by algo.py)
│   ├── scripts/
│   │   ├── segmenter.py               # Copy of template
│   │   └── ...
│   ├── model/
│   │   ├── model.pt                   # Best checkpoint
│   │   └── training.log               # Training log
│   └── prediction_validation/         # Fold validation predictions
│
├── fold_predictions/              # Reorganized validation predictions
│   └── fold0/ through fold4/
├── ensemble_output/               # Test set ensemble predictions
└── performance_metrics.csv        # Final metrics table
```

## Key Source Files

| File | Location | What it does |
|------|----------|-------------|
| `algo.py` | `algorithm_templates/segresnet/scripts/` | `fill_template_config()` — merges data stats + your params → hyper_parameters.yaml |
| `segmenter.py` | `algorithm_templates/segresnet/scripts/` | `Segmenter` class — all training, validation, inference logic |
| `utils.py` | `algorithm_templates/segresnet/scripts/` | `auto_adjust_network_settings()` — GPU-aware param scaling |
| `train.py` | `algorithm_templates/segresnet/scripts/` | Fire CLI entry point — calls `segmenter.run()` |
| `train.py` | `training/roi_train2/` | Your entry point — loads configs, sets up run dir, calls AutoRunner |
