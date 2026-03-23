# Performance Metrics Computation Guide

## Overview

This guide explains how to compute comprehensive performance metrics (sensitivity, specificity, FPR, FNR, etc.) for your segmentation model across different data splits (test, validation, training).

## Metric Definitions

All metrics treat:
- **Label 1 = Lesion (negative)**
- **Label 2 = Rim/PRL (positive)**

From the confusion matrix (TP, FP, TN, FN):

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Sensitivity** | TP / (TP+FN) | Ability to detect rims when present |
| **Specificity** | TN / (TN+FP) | Ability to correctly identify lesion areas |
| **FPR** | FP / (FP+TN) | 1 - Specificity; false alarm rate |
| **FNR** | FN / (TP+FN) | 1 - Sensitivity; miss rate |
| **Precision** | TP / (TP+FP) | Quality of positive predictions |
| **Accuracy** | (TP+TN) / Total | Overall correctness |
| **F1 Score** | 2×(Prec×Sens)/(Prec+Sens) | Harmonic mean of precision and sensitivity |

## Current Status (Test Set Analysis)

From `/media/smbshare/srs-9/prl_project/training/roi_train2/run2/ensemble_output`:

```
AGGREGATED METRICS (197 test cases):
  Sensitivity:  0.6917  ← Model detects rims 69% of the time
  Specificity:  0.9678  ← Model correctly identifies lesion areas 97% of the time
  FPR:          0.0322  ← False alarm rate is only 3.2%
  FNR:          0.3083  ← Misses 31% of rims (false negatives)
  Precision:    0.4289  ← When model predicts rim, it's correct 43% of the time
  F1 Score:     0.5295  ← Balanced measure of performance
```

**Key Insight:** The model is very good at NOT falsely identifying rims (high specificity), but struggles to detect all rims that are actually present (lower sensitivity). This suggests the model is conservative—it's better at avoiding false positives than catching all true positives.

## Using the Metrics Script

### For Test Set (Current)
```bash
python scripts/compute_performance_metrics.py \
  /media/smbshare/srs-9/prl_project/training/roi_train2/run2/datalist_xy20_z2.json \
  /media/smbshare/srs-9/prl_project/training/roi_train2/run2/ensemble_output \
  --data-root /media/smbshare/srs-9/prl_project/data \
  --test-only
```

### For All Splits (When Validation Predictions Exist)
```bash
python scripts/compute_performance_metrics.py \
  /path/to/datalist.json \
  /path/to/inference_root \
  --data-root /path/to/dataroot
```

### Save to CSV for Analysis
```bash
python scripts/compute_performance_metrics.py \
  /path/to/datalist.json \
  /path/to/inference_root \
  --data-root /path/to/dataroot \
  --output-csv results.csv
```

## Computing Metrics for Validation Folds

Currently, **you don't have inference results for validation folds** because MONAI Auto3DSeg doesn't automatically save per-case predictions during training.

To compute validation metrics per fold, you have two options:

### Option A: Generate Validation Predictions During Training (Best)

Modify the training script to save predictions on the validation set for each fold:

1. After training completes for each fold, run inference on the fold's validation data
2. Save predictions to a standardized location (e.g., `run2/fold_predictions/fold{N}/`)
3. Run the metrics script on each fold

```python
# Pseudo-code for what needs to be added to train.py
for fold_num in range(N_FOLDS):
    # Train model on folds except fold_num
    # Load trained model
    # Run inference on fold_num validation data
    # Save predictions to: run_dir / f"fold_predictions/fold{fold_num}/"
```

### Option B: Compute from Existing Training Metrics

For now, you can:
1. Use the MLflow metrics (sensitivity via val/acc_1, specificity approximation)
2. Combine with the test set results to understand fold-specific performance

**Trade-off:** Option B is faster but less detailed; Option A gives you the full confusion matrix per fold.

## Interpreting Results by Case Type

The script automatically categorizes cases:

| Type | Meaning | Metrics |
|------|---------|---------|
| **Lesion** | No rim present (lesion-only) | Sensitivity/Precision = N/A; Specificity = 0 or 1 |
| **PRL** | Rim present (PRL case) | All metrics defined |
| **Mixed** | Both rim and non-rim lesion regions | All metrics defined |

Cases marked "Lesion" with high accuracy are good—the model correctly avoided false positives.

## Next Steps

1. **Investigate Sensitivity Issues**: Why is sensitivity only 69%?
   - Are certain types of rims harder to detect?
   - Do larger rims (Fold 4) have different sensitivity?

2. **Per-Fold Analysis**: Once you generate validation predictions, run:
   ```bash
   python scripts/compute_performance_metrics.py \
     datalist.json \
     fold_predictions/fold0 \
     --data-root /path/to/dataroot
   ```
   for each fold to see if Fold 4 has systematically different metrics.

3. **Thresholding**: Current metrics use a hard threshold (probability > 0.5). Consider:
   - ROC curves: How does sensitivity/specificity change with threshold?
   - Optimal operating point: Find threshold that maximizes your desired metric

## Scripts Available

- `analyze_mlflow_runs.py` - MLflow metrics across folds (training loss, val_acc)
- `diagnose_fold_differences.py` - Analyze data distribution and characteristics per fold
- `compute_performance_metrics.py` - Detailed performance metrics (this guide)
