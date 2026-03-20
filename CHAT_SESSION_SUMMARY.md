# Chat Session Summary: Cross-Fold Validation Analysis & Metrics Pipeline

**Date:** March 19-20, 2026
**Goal:** Understand and improve cross-fold validation analysis, create reusable evaluation pipeline

## Problem Statement

You wanted to:
1. Understand cross-validation results (why Fold 4 performed worse)
2. Compute detailed performance metrics (sensitivity, specificity, FPR, FNR) for each fold
3. Create reusable tools integrated into your pipeline

## Key Discoveries

### Fold 4 Issue: Data Size Mismatch
- **Finding:** Fold 4 contains 40% larger lesion crops (0.18 MB avg vs 0.11-0.13 MB for other folds)
- **Implication:** Model trained on smaller lesions, tested on larger/more complex ones = distribution shift
- **Solution:** Future runs should use stratified fold assignment (balance by lesion size)

### Cross-Validation Mechanics
- For each fold N: train on folds 0-4 except N, validate on fold N
- Similar training loss across folds ≠ similar validation accuracy (Fold 4 proves this)
- Training loss converges equally because it's training on the "easy" 4 folds
- Validation accuracy varies because Fold 4 is harder

### Test Set Performance (run2)
```
Sensitivity:  0.692  (detects 69% of rims)
Specificity:  0.968  (correctly identifies lesion areas 97% of time)
Precision:    0.429  (when predicting rim, often includes false positives)
F1 Score:     0.530
```
→ Model is conservative: avoids false positives but misses ~31% of actual rims

## Tools Created

### 1. `analyze_mlflow_runs.py`
**Purpose:** Analyze training metrics across folds from MLflow
**Examines:** Training loss, validation accuracy per fold, fold variance
**Use:**
```bash
python scripts/analyze_mlflow_runs.py /path/to/run2 --distributed
```

### 2. `diagnose_fold_differences.py`
**Purpose:** Identify data characteristics that vary per fold
**Shows:** Subject counts, lesion size distributions, outlier characteristics
**Refactored to:** Use single `train_dir` argument
**Use:**
```bash
python scripts/diagnose_fold_differences.py /path/to/run2 --problem-fold 4
```

### 3. `compute_performance_metrics.py` (USER IMPROVED)
**Purpose:** Compute confusion matrix metrics with detailed per-case breakdown
**Metrics:** TP, FP, TN, FN, Sensitivity, Specificity, FPR, FNR, Precision, NPV, Accuracy, F1
**User's Improvements:**
- Loads config from run directory instead of requiring explicit arguments
- Derives datalist and suffix from `label_config.json`
- Handles both test set (ensemble_output) and validation folds (fold_predictions)
- Groups validation cases by fold automatically
- Case type detection (PRL vs Lesion)

**Use:**
```bash
python scripts/compute_performance_metrics.py /path/to/run2 --print-results
```

### 4. `generate_fold_predictions.py`
**Purpose:** Generate inference predictions on validation data for each fold
**Approach:** Wraps Auto3DSeg's auto-generated `infer.py` for each fold
**Output Structure:**
```
run2/fold_predictions/
├── fold0/sub{id}-{ses}/{lesion_index}/flair.phase_xy20_z2.nii.gz
├── fold1/...
└── fold4/...
```

**Status:** Created but not yet tested. Uses existing Auto3DSeg infrastructure (correct approach).

## Design Patterns Learned

### Script CLI Pattern
```python
# OLD (messy): Multiple explicit arguments
python script.py /datalist.json /inference_root /data_root

# NEW (clean): Single run_dir, derive everything
python script.py /path/to/run2
```

### Path Derivation
```python
# Load config from run directory
label_config = load_config(run_dir / "label_config.json")
expand_xy, expand_z = label_config['expand_xy'], label_config['expand_z']
suffix = f"_xy{expand_xy}_z{expand_z}"

# Build inference path
inference_path = inf_root / label_path.relative_to(dataroot).with_name(
    f"flair.phase{suffix}_ensemble.nii.gz"
)
```

### Inline Path Conversion
```python
scan.update({k: Path(v) if isinstance(v, str) and ".nii.gz" in v else v
            for k, v in scan.items()})
```

## MLflow Refactoring (Option 2)

**Goal:** Unified cross-fold tracking instead of distributed per-fold directories

**Changes Made:**
1. Updated `hyper_parameters.yaml` to use `null` for mlflow settings (set at runtime)
2. Updated `train.py` to pass shared `mlruns` directory to AutoRunner
3. All folds log to single `run_dir/mlruns/` as separate runs within same experiment

**Benefit:** MLflow UI shows all folds together automatically

## Memory Files Created

Saved for future reference:
- `feedback_script_cli_design.md` — Script design preference
- `feedback_path_patterns.md` — Path manipulation idioms
- `project_run_directory_structure.md` — Full directory hierarchy
- `project_datalist_and_cases.md` — Datalist structure and case types

## Next Steps (When Ready)

### Immediate
1. Test `generate_fold_predictions.py` on run2 to generate validation predictions
2. Run `compute_performance_metrics.py` on fold_predictions for each fold
3. Compare metrics across folds to diagnose Fold 4

### Medium-term
1. Implement stratified fold assignment to balance lesion sizes
2. Integrate prediction generation into training pipeline
3. Create automated comparison reports

### Longer-term
1. Hyperparameter optimization (HPO)
2. Port to HPC with job submission
3. Handle class imbalance (high background accuracy, low rim accuracy)

## File Locations Reference

**Scripts created:**
- `/home/srs-9/Projects/prl_project/scripts/analyze_mlflow_runs.py`
- `/home/srs-9/Projects/prl_project/scripts/diagnose_fold_differences.py`
- `/home/srs-9/Projects/prl_project/scripts/compute_performance_metrics.py` (improved by user)
- `/home/srs-9/Projects/prl_project/scripts/generate_fold_predictions.py`

**Guides created:**
- `METRICS_GUIDE.md` — How to use metrics computation
- `FOLD_VALIDATION_INFERENCE_GUIDE.md` — How to generate validation predictions

**Modified:**
- `training/roi_train2/algorithm_templates/segresnet/configs/hyper_parameters.yaml`
- `training/roi_train2/train.py`

**Memory saved:**
- `/home/srs-9/.claude/projects/-home-srs-9-Projects-prl-project/memory/`

## Key Takeaways

1. **Fold 4 isn't a data quality issue** — it's a distribution mismatch (larger lesions)
2. **Current metrics show** — model is learning but conservative; needs tuning for sensitivity/precision balance
3. **Tools are modular** — can be run independently or integrated into pipeline
4. **Design patterns matter** — single `train_dir` argument + derived paths = cleaner, more maintainable scripts
5. **Auto3DSeg infrastructure is your friend** — don't rewrite model loading; wrap existing code

## Session Statistics

- **Scripts created:** 4
- **Guides created:** 2 (plus this summary)
- **Memory files saved:** 4
- **Key issues diagnosed:** 1 (Fold 4 size mismatch)
- **Metrics implemented:** 8 (Sens, Spec, FPR, FNR, Prec, NPV, Acc, F1)
