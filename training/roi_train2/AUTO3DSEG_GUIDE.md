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

### Stage 1: Your configs → AutoRunner (two separate paths)

**`train.py`** loads configs and splits parameters into two paths:

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

- **`input` dict** → baked into `hyper_parameters.yaml` during config generation. Lists and complex types go here.
- **`set_training_params()`** → applied as CLI overrides at training time. Scalars only.

See "The Two Parameter Paths" in the Deep Dive section for the full explanation of why.

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
   - **This is the key line**: any `input` dict key that wasn't explicitly popped earlier gets injected into the final config. This is how `crop_ratios`, `roi_size`, etc. make it through.
7. **Writes** the final config to `segresnet_<fold>/configs/hyper_parameters.yaml`

**Important**: The `config.update(input_config)` is a **shallow merge** on a plain dict. For nested keys like `loss` (which is itself a dict), passing `loss: {weight: [1,1,5]}` would REPLACE the entire loss dict, not merge into it. However, when the dict is written to YAML via ConfigParser (lines 276-295), the `#` notation in key names IS supported as nested key access (e.g., `loss#weight` would set `loss.weight`).

### Stage 3: BundleAlgo.train() — Training launch

After config generation, `BundleAlgo.train(train_params=...)` launches training:

1. Converts `train_params` to CLI arguments via `check_and_set_optional_args()`
   - **Scalars**: converted to `--key=value`
   - **Lists**: converted to Fire-compatible format `--key=[v1,v2,v3]`
   - **Dicts**: **REJECTED** with `ValueError("Nested dict is not supported")`
2. Runs `python scripts/train.py run --config_file=configs/hyper_parameters.yaml --key1=val1 ...`
3. Python Fire parses CLI args → passed as `config_dict` override to `Segmenter`

**Implication**: You can pass scalars and lists through `train_param`, but NOT nested dicts (like `loss`). The loss config must be handled separately if you want to modify it.

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
| `max_samples_per_class` | 10 x num_epochs | Max cached voxel indices per class |
| `crop_add_background` | true | Include background as a crop class |

**How `crop_mode="ratio"` works**: `RandCropByLabelClassesd` picks a random class according to `ratios` (or uniform if null), then centers the crop on a random voxel of that class. With 3 output classes and null ratios, each class has ~33% chance. With `[1, 1, 4]`, rim gets `4/6 = 67%` chance of being the center class.

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
| `loss.weight` | none | **Per-class weights** -- not set by default. Would be `[1.0, 1.0, 5.0]` to emphasize rim |

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
- `batch_size` -> 2 initially
- Swap: `num_crops_per_image` = 2, `batch_size` = 1
- `num_epochs` = 500 // min(3, 2) = 250 effective epochs
- Each step: 2 crops from 1 image (more spatial diversity, half the epochs)

With `num_images_per_batch=1` (current):
- `num_crops_per_image` = 1, `batch_size` = 1
- `num_epochs` = 500 // 1 = 500 (unchanged)

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

## What You Can vs. Can't Control

### Via `input` dict (→ `fill_template_config()` → baked into hyper_parameters.yaml):
- **Lists**: `roi_size`, `crop_ratios`, `resample_resolution`
- **Scalars**: everything below also works here, but scalars are simpler via `set_training_params()`

### Via `set_training_params()` (→ CLI overrides at training time, scalars only):
- `learning_rate`, `num_epochs`, `num_warmup_epochs`, `num_epochs_per_validation`
- `num_images_per_batch`, `early_stopping_fraction`, `num_workers`
- `auto_scale_batch`, `auto_scale_roi`, `auto_scale_filters` (booleans)

### Cannot set via either path (nested dicts):
- `loss` (dict with `_target_`, params) -- rejected by `check_and_set_optional_args()`
- `optimizer` (dict) -- same issue
- `network` (dict) -- same issue

### Workaround for nested params:
Modify `train.py` to run AutoRunner in phases: generate configs first, patch the YAML, then train. Or use the `#` notation (e.g., `network#init_filters`) which ConfigParser handles as nested keys during YAML writing -- but may fail at the Fire CLI parsing stage.

**Note**: `train.py` auto-routes list-valued params from `train_param` to `input_dict` via the `isinstance(..., list)` check, so you can put everything in `train_param` in `monai_config.jsonc` and the routing happens automatically.

## Key Source Files

| File | Location | What it does |
|------|----------|-------------|
| `algo.py` | `algorithm_templates/segresnet/scripts/` | `fill_template_config()` -- merges data stats + your params -> hyper_parameters.yaml |
| `segmenter.py` | `algorithm_templates/segresnet/scripts/` | `Segmenter` class -- all training, validation, inference logic |
| `utils.py` | `algorithm_templates/segresnet/scripts/` | `auto_adjust_network_settings()` -- GPU-aware param scaling |
| `train.py` | `algorithm_templates/segresnet/scripts/` | Fire CLI entry point -- calls `segmenter.run()` |
| `my_autorunner.py` | `training/roi_train2/` | Your wrapper -- loads configs, creates AutoRunner |
| `train.py` | `training/roi_train2/` | Your entry point -- loads configs, sets up run dir, calls AutoRunner |

---

## Deep Dive: AutoRunner Internals

Detailed notes on AutoRunner behavior discovered through debugging and source code analysis. The sections above give the big picture; this section covers the specifics you'll need when things go wrong or when you want to push beyond the standard workflow.

### The Two Parameter Paths (input dict vs set_training_params)

This is the single most important architectural detail. There are **two completely separate paths** for getting parameters into training, and they have different capabilities:

**Path 1: `input` dict** (passed to `AutoRunner(input={...})`)
- Written to an internal YAML file
- Read by `algo.fill_template_config()` during the algo_gen phase
- Merged into `hyper_parameters.yaml` via `config.update(input_config)` at algo.py line 271
- **Supports all types**: scalars, lists, dicts
- Parameters are **baked into the YAML** before training starts

**Path 2: `set_training_params()`**
- Stored by AutoRunner and passed to `BundleAlgo.train(train_params=...)` at training time
- Converted to CLI arguments via `check_and_set_optional_args()` in bundle_gen.py
- Applied as **runtime overrides** on top of `hyper_parameters.yaml`
- **Scalars only**: lists get mangled by Python Fire (e.g., `[1,1,4]` becomes the string `"1,1,4"`), dicts are rejected with `ValueError`

**Why this matters**: If you put `crop_ratios: [1, 1, 4]` in `set_training_params()`, it flows through `BundleAlgo.train()` → CLI args → Fire parsing, where `[1,1,4]` becomes a string. The segmenter then crashes because it expects a list. The fix is to route list-valued params through the `input` dict instead.

Our `train.py` handles this automatically:
```python
# List-valued params go in input_dict (path 1), scalars stay in train_param (path 2)
for key in list(train_param):
    if isinstance(train_param[key], list):
        input_dict[key] = train_param.pop(key)
```

### Phase Caching (cache.yaml)

AutoRunner tracks which phases have completed in `{work_dir}/cache.yaml`:

```yaml
{analyze: true, datastats: /path/to/datastats.yaml, algo_gen: true, train: false}
```

When AutoRunner starts, it reads this file and skips phases that are already marked `true`. The `train` flag is only set to `true` after **all** folds finish training (at auto_runner.py line 891). If the process is killed mid-training, `train` stays `false`, so AutoRunner knows to re-enter the training phase on the next run.

**When caching bites you**: If you change parameters and want to regenerate `hyper_parameters.yaml`, AutoRunner will skip algo_gen because `algo_gen: true` in the cache. You must either:
- Delete `cache.yaml` (AutoRunner re-evaluates everything from scratch)
- Delete the entire work_dir
- Pass `not_use_cache=True` to the AutoRunner constructor

The cache also has a self-healing check: if `algo_gen: true` but no `algo_object.pkl` files exist, it resets `algo_gen` to `false`. Similarly for `train`.

### Fold Completion Detection

When AutoRunner re-enters the training phase, it decides which folds to skip vs retrain using `import_bundle_algo_history()` (in `monai.apps.auto3dseg.utils`):

```
For each segresnet_N/ directory:
  1. Look for algo_object.pkl — if missing, fold is invisible (skipped entirely)
  2. Read best_metric from the pkl's metadata
  3. If best_metric is None, fall back to algo.get_score()
     → reads {ckpt_path}/progress.yaml
     → returns the last entry's best_avg_dice_score
  4. is_trained = (best_metric is not None)
```

Then at auto_runner.py line 876-883:
```python
skip_algos = [h[AlgoKeys.ID] for h in history if h[AlgoKeys.IS_TRAINED]]
history = [h for h in history if not h[AlgoKeys.IS_TRAINED]]
```

Only folds with `is_trained=False` are passed to `_train_algo_in_sequence()`.

**The problem with killed jobs**: When a fold is killed mid-training, it still has:
- `algo_object.pkl` (created before training starts) — but with `best_metric: None` in metadata
- `progress.yaml` with incremental entries including `best_avg_dice_score`

So `get_score()` succeeds and returns a score → `is_trained=True` → fold is skipped, even though it didn't finish. The partially-trained model is treated as the final result.

**Fix for killed folds**: Delete `progress.yaml` from the fold's `model/` directory. This causes `get_score()` to raise `FileNotFoundError` (caught by `except BaseException: pass`), so `best_metric` stays `None` → `is_trained=False` → fold is retrained. The fold will **retrain from epoch 0**, not resume from the checkpoint, because `continue` is not set in `hyper_parameters.yaml`.

### Per-Fold Output Files

Each `segresnet_N/` fold directory contains these files relevant to completion tracking:

```
segresnet_N/
├── algo_object.pkl              # Created during algo_gen phase (before training)
│                                # Contains: algo_bytes, template_path, best_metric
│                                # best_metric is None until training completes and
│                                # _train_algo_in_sequence() writes it back with the score
├── configs/
│   └── hyper_parameters.yaml    # Generated config (all resolved parameters)
├── model/
│   ├── model.pt                 # Best checkpoint (overwritten when metric improves)
│   ├── model_final.pt           # Latest checkpoint (overwritten every epoch)
│   ├── progress.yaml            # Incremental log: list of dicts with best_avg_dice_score,
│   │                            # epoch, time, etc. Written every num_epochs_per_saving epochs.
│   │                            # This is what get_score() reads.
│   ├── accuracy_history.csv     # Per-epoch metrics (train loss, val dice per class)
│   └── training.log             # Text log of training progress
└── scripts/
    └── ...                      # Copies of algorithm template scripts
```

**Completion lifecycle**:
1. `algo_gen` phase: creates `algo_object.pkl` with `best_metric: None`
2. Training: writes `model.pt`, `progress.yaml`, `accuracy_history.csv` incrementally
3. Training completes: `_train_algo_in_sequence()` calls `algo.get_score()`, then re-pickles `algo_object.pkl` with the actual `best_metric` value

If a job is killed between steps 2 and 3, the pkl still has `best_metric: None` but `progress.yaml` has scores — this is the state that fools the fallback logic.

### The 250 vs 500 Epoch Discrepancy

When `auto_scale_allowed=true` and `crop_mode="ratio"`, segmenter.py swaps `batch_size` and `num_crops_per_image`, then halves epochs. This happens at runtime inside segmenter.py, **not** in hyper_parameters.yaml. So:

- `hyper_parameters.yaml` says `num_epochs: 500`, `batch_size: 1` (or 2 after algo.py auto-scaling)
- segmenter.py at runtime: if batch_size was auto-scaled to 2, it becomes `num_crops_per_image=2, batch_size=1, num_epochs=250`
- The progress bar shows 250 epochs, `accuracy_history.csv` has ~249 entries
- But the total gradient steps are equivalent (250 epochs × 2 crops = 500 × 1 crop)

The `num_crops_per_image` value in `hyper_parameters.yaml` reflects the post-swap value (written by algo.py's auto-scaling), so you can check it there to understand what effective epoch count to expect.

### AutoRunner's Built-in NNI HPO

AutoRunner supports grid search via NNI (Neural Network Intelligence) with `hpo=True` and `set_nni_search_space()`. However, NNI HPO goes through the same `BundleAlgo.train()` → CLI args path, so it has the **same list-mangling problem** with Python Fire. It also adds NNI as a dependency and manages trial lifecycle internally, giving you less control over individual runs.

For our use case (9-run grid, LSF job arrays, need independent restartable runs), the separate-run-directories approach is simpler and more robust.

### Checkpoint Resumption Within a Fold

segmenter.py supports resuming from a checkpoint via two config keys:
- `continue: true` — tells `checkpoint_load()` to restore `start_epoch` and `best_metric` from the checkpoint
- `start_epoch` — the epoch to resume from

Neither is set by default in `hyper_parameters.yaml`. If you wanted to resume a killed fold from its last checkpoint rather than retraining from scratch, you'd need to manually add `continue: true` to that fold's `hyper_parameters.yaml` before rerunning. The existing `model.pt` (best checkpoint) and `model_final.pt` (latest checkpoint) would then be loaded.
