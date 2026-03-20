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
                            │         ├── reads input_config (your train_param values)
                            │         ├── auto-adjusts network settings (roi, filters, batch)
                            │         └── writes hyper_parameters.yaml (per fold)
                            │
                            └──→ BundleAlgo.train() ──→ segmenter.py
                                      │                     │
                                      │                     ├── reads hyper_parameters.yaml
                                      └── CLI overrides ──→ ├── builds transforms, model, loss
                                                            └── runs training loop
```

## Parameter Flow in Detail

### Stage 1: Your configs → AutoRunner

**`my_autorunner.py`** loads both config files, extracts `train_param`, and passes it:

```python
train_param = monai_config['train_param']
runner = AutoRunner(work_dir=..., algos=["segresnet"], input={...})
runner.set_training_params(train_param)
runner.run()
```

AutoRunner writes the combined input config (modality, datalist path, dataroot, **plus all your train_param values**) to an internal YAML file. This becomes the `data_list_file` for the algorithm.

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
   - **This is the key line**: any train_param key that wasn't explicitly popped earlier gets injected into the final config. This is how `crop_ratios`, `num_epochs`, `learning_rate`, etc. make it through.
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

## What You Can vs. Can't Control via train_param

### Safe to set (scalars/lists -- flow cleanly):
- `learning_rate`, `num_epochs`, `num_warmup_epochs`, `num_epochs_per_validation`
- `roi_size` (list), `crop_ratios` (list), `num_images_per_batch`
- `early_stopping_fraction`, `num_workers`
- `auto_scale_batch`, `auto_scale_roi`, `auto_scale_filters` (booleans)

### Cannot set via train_param (nested dicts fail at CLI stage):
- `loss` (dict with `_target_`, params) -- requires hyper_parameters.yaml post-processing
- `optimizer` (dict) -- same issue
- `network` (dict) -- same issue

### Workaround for nested params:
Modify `train.py` to run AutoRunner in phases: generate configs first, patch the YAML, then train. Or use the `#` notation (e.g., `network#init_filters`) which ConfigParser handles as nested keys during YAML writing -- but may fail at the Fire CLI parsing stage.

## Key Source Files

| File | Location | What it does |
|------|----------|-------------|
| `algo.py` | `algorithm_templates/segresnet/scripts/` | `fill_template_config()` -- merges data stats + your params -> hyper_parameters.yaml |
| `segmenter.py` | `algorithm_templates/segresnet/scripts/` | `Segmenter` class -- all training, validation, inference logic |
| `utils.py` | `algorithm_templates/segresnet/scripts/` | `auto_adjust_network_settings()` -- GPU-aware param scaling |
| `train.py` | `algorithm_templates/segresnet/scripts/` | Fire CLI entry point -- calls `segmenter.run()` |
| `my_autorunner.py` | `training/roi_train2/` | Your wrapper -- loads configs, creates AutoRunner |
| `train.py` | `training/roi_train2/` | Your entry point -- loads configs, sets up run dir, calls AutoRunner |
