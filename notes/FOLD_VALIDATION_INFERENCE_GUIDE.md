# Generating Validation Predictions per Fold

## Overview

This guide explains how to generate inference predictions on validation data for each fold, which enables you to compute detailed performance metrics (sensitivity, specificity, FPR, FNR, etc.) for each fold separately.

## Architecture

The approach leverages MONAI Auto3DSeg's auto-generated inference infrastructure:

```
For each fold:
  1. Load the trained model (segresnet_N/model/model.pt)
  2. Use the auto-generated infer.py infrastructure
  3. Run inference on that fold's validation data
  4. Save predictions organized by fold
  5. Later compute metrics from predictions + ground truth labels
```

## Files

### Scripts

- **`scripts/generate_fold_predictions.py`** - Main script to generate validation predictions
  - Uses the auto-generated Auto3DSeg inference code
  - Runs for all folds or a specific fold
  - Saves predictions organized by fold

- **`compute_performance_metrics.py`** - Computes detailed performance metrics
  - Works with the generated predictions
  - Computes sensitivity, specificity, FPR, FNR, etc.

### Auto-Generated Infrastructure (Auto3DSeg)

Each fold has auto-generated scripts at:
```
run2/segresnet_N/scripts/
  ├── infer.py              # Inference entry point
  ├── validate.py           # Validation entry point
  ├── segmenter.py          # Core segmentation class
  ├── algo.py               # Algorithm configuration
  └── utils.py              # Utilities
```

These handle:
- Model loading
- Data preprocessing/transforms
- Sliding window inference
- Output formatting

## Usage

### Generate validation predictions for all folds

```bash
python scripts/generate_fold_predictions.py \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2 \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2/datalist_xy20_z2.json
```

This will:
1. Loop through folds 0-4
2. Run inference on each fold's validation data
3. Save predictions to `run2/fold_predictions/fold{N}/`

### Generate for a specific fold

```bash
python scripts/generate_fold_predictions.py \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2 \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2/datalist_xy20_z2.json \
    --fold 4
```

### Use custom output directory

```bash
python scripts/generate_fold_predictions.py \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2 \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2/datalist_xy20_z2.json \
    --output-dir /custom/path/fold_predictions
```

## Output Structure

After running inference:

```
run2/fold_predictions/
├── fold0/
│   └── sub{id}-{ses}/
│       └── {lesion_index}/
│           └── flair.phase_xy20_z2.nii.gz
├── fold1/
│   └── ...
├── fold2/
│   └── ...
├── fold3/
│   └── ...
└── fold4/
    └── ...
```

## Computing Performance Metrics

After generating predictions, compute metrics for each fold:

```bash
# For fold 0
python scripts/compute_performance_metrics.py \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2/datalist_xy20_z2.json \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2/fold_predictions/fold0 \
    --data-root /media/smbshare/srs-9/prl_project/data

# For fold 4
python scripts/compute_performance_metrics.py \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2/datalist_xy20_z2.json \
    /media/smbshare/srs-9/prl_project/training/roi_train2/run2/fold_predictions/fold4 \
    --data-root /media/smbshare/srs-9/prl_project/data
```

This will output:
- Sensitivity, Specificity, FPR, FNR
- Precision, NPV, Accuracy, F1 Score
- Per-case breakdown
- Identification of outlier cases

## Integration into Training Pipeline

To integrate prediction generation into your training pipeline:

### Option 1: Call after training completes

Add to `train.py` after `runner.run()`:

```python
# After training completes
import subprocess
import sys

result = subprocess.run([
    sys.executable,
    "scripts/generate_fold_predictions.py",
    str(work_dir),
    str(datalist_file),
])

if result.returncode != 0:
    print("Warning: Fold prediction generation failed")
```

### Option 2: Create a separate post-processing script

```bash
#!/bin/bash
# post_train.sh - Run after training

RUN_DIR=$1
DATALIST=$2
DATA_ROOT=$3

python scripts/generate_fold_predictions.py "$RUN_DIR" "$DATALIST"
python scripts/compute_performance_metrics.py "$DATALIST" "$RUN_DIR/fold_predictions/fold0" --data-root "$DATA_ROOT"
python scripts/compute_performance_metrics.py "$DATALIST" "$RUN_DIR/fold_predictions/fold1" --data-root "$DATA_ROOT"
# ... repeat for each fold
```

## Why This Approach

1. **Leverages existing code**: Uses Auto3DSeg's battle-tested inference infrastructure
2. **Consistent with training**: Uses the same preprocessing and model loading as training
3. **Flexible**: Can run post-training or integrated into the pipeline
4. **Reusable**: The metrics script works with any inference results
5. **Fold-specific analysis**: Enables debugging performance issues per fold (like Fold 4)

## Next Steps

1. Run `generate_fold_predictions.py` on run2
2. For each fold, run `compute_performance_metrics.py`
3. Compare metrics across folds to understand performance variation
4. Investigate fold 4's larger file sizes and how that affects metrics
5. For future runs, integrate prediction generation into your training pipeline

## Troubleshooting

### "Segmenter script not found"
- Ensure Auto3DSeg generated the scripts (they should be in segresnet_N/scripts/)
- Check that run2 structure is intact

### "Config file not found"
- Verify the configs exist in segresnet_N/configs/hyper_parameters.yaml
- This is auto-generated by Auto3DSeg

### Inference is slow
- The Auto3DSeg inference uses sliding window approach to handle memory
- Large lesion crops (like Fold 4) will be slower
- This is normal and expected

## Related Scripts

- `analyze_mlflow_runs.py` - Analyze training metrics across folds
- `diagnose_fold_differences.py` - Identify data characteristics per fold
- `compute_performance_metrics.py` - Compute detailed performance statistics
