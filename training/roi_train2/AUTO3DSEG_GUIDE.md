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

---

## Deep Parameter Reference

This section documents every parameter in `hyper_parameters.yaml` — how it's actually set, what can override it, and what it controls at runtime.

### How to read this section

Each parameter has an **Authority** — who wins if multiple sources set a value:
- **Template default** — the value in the unfilled `algorithm_templates/segresnet/configs/hyper_parameters.yaml`
- **algo.py** — set during `fill_template_config()` from data stats; overwrites the template
- **input_config** — your values from `AutoRunner(input={...})`; merged via `config.update(input_config)` in algo.py line 271, overwriting algo.py's auto-computed values
- **segmenter.py runtime** — final adjustments at training start; always wins

---

### Batch Size / Crops (most confusing chain)

#### `num_images_per_batch`
**Authority: template default = 1 (used as initial `batch_size` only)**

The template contains:
```yaml
batch_size: '@num_images_per_batch'
num_images_per_batch: 1
```
`@num_images_per_batch` is a ConfigParser reference that resolves at load time — so `batch_size` starts as whatever `num_images_per_batch` is. However, after this initial resolution, `num_images_per_batch` is never read again by any script. **Setting it in your input_config has no effect** because:
1. The `@` reference is resolved once into `batch_size`
2. If `auto_scale_batch=true`, segmenter.py immediately overwrites `batch_size` at runtime anyway

#### `batch_size`
**Authority: segmenter.py runtime (when `auto_scale_batch=true`)**

Flow:
1. Starts as `@num_images_per_batch` (from template)
2. algo.py may compute a value via `auto_adjust_network_settings()`, but this is used during config generation, not at runtime
3. **At runtime** (segmenter.py line 644-657): if `auto_scale_allowed=true` and `auto_scale_batch=true`, `auto_adjust_network_settings()` runs again and sets `batch_size = int(1.1 * gpu_factor_init)` based on GPU memory and roi_size
4. **Then** (segmenter.py lines 1195-1201): if `crop_mode="ratio"`:
   ```python
   num_crops_per_image = batch_size   # batch_size moves here
   batch_size = 1                     # always 1 in ratio mode
   ```

To control crops-per-image, **set `batch_size` directly and disable `auto_scale_batch`**:
```json
"auto_scale_batch": false,
"batch_size": 4
```
This gives `num_crops_per_image=4` after the swap. Setting `num_images_per_batch` or `num_crops_per_image` directly does nothing useful.

#### `num_crops_per_image`
**Authority: segmenter.py runtime (derived from batch_size swap)**

- Default in `parse_input_config()`: 1 (line 876)
- **Overwritten unconditionally** at runtime (lines 1197-1201) when `auto_scale_allowed=true`:
  - `crop_mode="ratio"`: `num_crops_per_image = batch_size`, then `batch_size = 1`
  - Otherwise: `num_crops_per_image = 1`
- Setting this in your input_config is pointless — it will be overwritten

**Downstream effects of num_crops_per_image = N:**
```
num_epochs          = num_epochs // min(3, N)    # halved at N=2, one-third at N≥3
num_warmup_epochs   = max(3, warmup // N)
num_epochs_per_saving = max(1, saving // N)
num_epochs_per_validation = max(1, validation // N)
RandCropByLabelClassesd(num_samples=N, ...)  # N patches per image per step
```

---

### Auto-Scale Flags

All four flags only apply when `auto_scale_allowed=true` (master switch).

#### `auto_scale_allowed`
**Authority: template default = true**

Master switch. If false, all `auto_scale_*` flags are ignored and runtime adjustments (including the batch/crop swap) are skipped. The only way to completely take manual control over batch_size and num_crops_per_image is to set this to false.

#### `auto_scale_batch`
**Authority: template default = true**

If true, segmenter.py recalculates `batch_size` at runtime based on GPU memory:
```python
gpu_factor_init = gpu_memory_fraction * base_numel / roi_size.prod()
batch_size = int(1.1 * gpu_factor_init)
```
On your V100s with `roi_size=[44,44,8]`, this computed `batch_size=2`, giving `num_crops_per_image=2` for all runs in the stage2 sweep regardless of what you set.

Set `"auto_scale_batch": false` if you want to control batch_size / num_crops_per_image yourself.

#### `auto_scale_roi`
**Authority: template default = false**

If true, roi_size is scaled UP to fill GPU memory (opposite of auto_scale_batch — bigger patches instead of more patches). Disabled when using a pretrained model.

#### `auto_scale_filters`
**Authority: template default = false**

If true, `init_filters` is increased when roi_size is small (since small ROIs leave spare GPU memory). Disabled when using a pretrained model.

---

### Cropping

#### `crop_mode`
**Authority: algo.py (auto-determined), overridable via input_config**

Set by algo.py based on roi_size vs image_size:
- `"ratio"` if any roi dimension < 0.8 × image dimension → `RandCropByLabelClassesd`
- `"rand"` otherwise → `RandSpatialCropd` (no class balancing)

For this project, roi_size=[44,44,8] is much smaller than image_size=[51,52,16], so `crop_mode` is always `"ratio"`.

#### `crop_ratios`
**Authority: input_config (no auto-computation)**

Per-class sampling weights for `RandCropByLabelClassesd`. Only active in `crop_mode="ratio"`.
- `null` (default): equal probability for each class
- `[1, 1, 4]`: 4x more crops centered on rim voxels

Stage 1 sweep showed this had no effect here because roi_size ≈ image_size — the crop always captures the whole lesion regardless of centering.

#### `cache_class_indices`
**Authority: segmenter.py runtime**

If null, auto-set to `True` when `cache_rate_train > 0`. Pre-computes which voxels belong to each class so `RandCropByLabelClassesd` doesn't scan every epoch. Only meaningful in `crop_mode="ratio"`.

#### `max_samples_per_class`
**Authority: segmenter.py default = 10 × num_epochs**

Limits cached voxels per class for `ClassesToIndicesd`. Only meaningful when `cache_class_indices=True`.

---

### Resampling

#### `resample`
**Authority: algo.py (auto-determined), overridable**

Auto-set to `True` if any case's spacing deviates more than ±50% from target spacing. For this project, spacings are very consistent, so `resample=False`.

#### `resample_resolution`
**Authority: algo.py (auto-computed from data stats), overridable**

Target voxel spacing. Even when `resample=False`, this value is used by `auto_adjust_network_settings()` to compute roi_size and gpu_factor. Auto-computed as median spacing by default. For this project: approximately `[0.8, 0.8, 0.8]` mm.

#### `resample_mode`
**Authority: input_config (not in hyper_parameters.yaml directly)**

Controls how algo.py computes `resample_resolution`. Options: `"auto"` (default), `"median"`, `"median10"`, `"ones"`. Only relevant if you want to override the auto-computed target spacing.

---

### Loss

#### `loss`
**Authority: input_config (can fully replace), template default is DiceCELoss**

The full loss config dict. When passed through input_config, it replaces the entire template loss dict (shallow merge). The `$expression` syntax is evaluated by ConfigParser before instantiation:
- `softmax: $not @sigmoid` → resolves to `softmax=True` when `sigmoid=False`
- `sigmoid: $@sigmoid` → mirrors the top-level `sigmoid` parameter
- `to_onehot_y: $not @sigmoid` → label one-hot encoding when using softmax

The loss is wrapped in `DeepSupervisionLoss` automatically for the SegResNetDS deep supervision heads.

To pass a custom loss including `weight`, use:
```json
"weight": "$torch.tensor([1.0, 1.0, 5.0]).cuda()"
```
Plain Python lists fail (`TypeError: cannot assign 'list' object to buffer 'weight'`) because PyTorch requires a Tensor. The `.cuda()` is required — without it you get a device mismatch error.

#### `sigmoid`
**Authority: algo.py (auto-determined from class_index)**

If class labels are non-overlapping (standard case), `sigmoid=False` → softmax + one-hot. If class indices overlap across classes (multi-label), `sigmoid=True` → independent sigmoid per class. For this project always `False`.

---

### Network

#### `blocks_down`
**Authority: algo.py (auto-determined from levels)**

Encoder depth profile. Computed from number of resolution levels:
- 5 levels: `[1, 2, 2, 4, 4]`
- 4 levels: `[1, 2, 2, 4]`
- 3 levels: `[1, 2, 4]` ← for this project (roi_size=[44,44,8])
- 2 levels: `[1, 3]`

#### `init_filters`
**Authority: algo.py / segmenter.py runtime (when `auto_scale_filters=true`)**

Base filter count (doubles at each encoder level). Default 32. Only auto-scaled up when `auto_scale_filters=true` and roi_size is small. For this project: 32.

#### `roi_size`
**Authority: algo.py (auto-computed), overridable via input_config**

Patch size for training crops and sliding-window inference. Overriding this is the primary lever for changing network capacity. For this project explicitly set to `[44, 44, 8]` to match the lesion ROI dimensions.

---

### Intensity Normalization

#### `normalize_mode`
**Authority: algo.py (auto-determined from modality)**

- MRI → `"meanstd"` (z-score normalization)
- CT → `"range"` (clip to `intensity_bounds`, scale to [-1,1])

#### `intensity_bounds`
**Authority: algo.py (auto-computed as 0.5th / 99.5th percentile)**

Only used in `"range"` / CT normalization. For MRI (meanstd), this is set but unused.

---

### Training Dynamics

#### `num_epochs`
**Authority: algo.py (auto-computed from dataset size), overridable — then adjusted at runtime**

algo.py formula: `clip(ceil(80000 / n_cases), 300, 1250)`. Can be overridden via input_config. Then at runtime, further reduced: `num_epochs = num_epochs // min(3, num_crops_per_image)`. So if you set 500 epochs and end up with `num_crops_per_image=2`, you get 250 effective epochs.

#### `learning_rate`
**Authority: input_config / template default = 2e-4**

No auto-adjustment. Referenced by `optimizer.lr: '@learning_rate'` so changing one changes both.

#### `early_stopping_fraction`
**Authority: input_config / template default = 0.001**

If validation metric improves less than this fraction over the last 10% of epochs (checked after the halfway point), training stops early.

#### `cache_rate`
**Authority: segmenter.py runtime (auto-computed from available RAM)**

If null, computed from system RAM and dataset size: `min(available_ram / (data_size + 50GB_os_reserve), 1.0)`. Train set gets priority. Values < 0.1 round to 0 (no caching). You can set an explicit value to override.

---

### Parameters That Look Settable But Are Ignored or Overridden

| Parameter | Why it doesn't do what you'd expect |
|-----------|-------------------------------------|
| `num_images_per_batch` | Resolves once into `batch_size` at config parse time; never read again. Overridden by `auto_scale_batch` anyway. |
| `num_crops_per_image` | Unconditionally overwritten by the batch_size↔crops swap at runtime (when `auto_scale_allowed=true`). |
| `batch_size` (when `auto_scale_batch=true`) | Recomputed from GPU memory, then moved to `num_crops_per_image` in ratio mode. |
| `num_epochs` | Divided by `min(3, num_crops_per_image)` at runtime. Set to 500, get 250 with 2 crops/image. |
| `intensity_bounds` | Unused for MRI (meanstd normalization). Only active for CT. |
| `num_warmup_epochs`, `num_epochs_per_saving`, `num_epochs_per_validation` | All divided by `num_crops_per_image` at runtime. |

---

### How to Actually Control num_crops_per_image

The only reliable way:
```json
"auto_scale_batch": false,
"batch_size": 2
```
With `auto_scale_batch=false`, the runtime recalculation is skipped, and `batch_size` stays as you set it. Then the `crop_mode="ratio"` swap gives `num_crops_per_image=2, batch_size=1`.

Do **not** set `num_images_per_batch` or `num_crops_per_image` directly — neither survives to training.
