
# PRL Segmentation and Classification Pipeline Specification (Revised)

## Overview

This document defines an implementation-agnostic specification for a pipeline to detect and classify Paramagnetic Rim Lesions (PRLs) from MRI data.

The pipeline should:

- Be modular, reproducible, and extensible
- Support both CLI and notebook workflows
- Clearly separate data staging, preprocessing, training, inference, and analysis
- Preserve full provenance of all derived data

---

## Core Goals

1. Stage MRI and lesion data from original source
2. Generate lesion-centered crops with tunable parameters
3. Train deep learning segmentation models (Auto3DSeg default)
4. Run inference on labeled and unlabeled subjects
5. Compute metrics where ground truth exists
6. Extract lesion-level features
7. Train/apply lesion-level classifiers
8. Produce final classifier-filtered PRL outputs

---

## Stable Identifiers

All pipeline stages must preserve:

- Subject ID: `(subid, sesid)`
- Lesion ID: `(subid, sesid, lesion_index)`

These identifiers must appear in all manifests, metrics, and outputs.

---

## Data Sources

### Original Data Root

```
ORIG_DATAROOT=/media/smbshare/3Tpioneer_bids
```

Structure:

```
sub-ms{subid}/ses-{sesid}
```

Session mapping:

```
subject-sessions.csv
```

---

## Subject Categories

### 1. Ground-truth subjects
- Full voxelwise PRL labels
- Used for segmentation training and evaluation
- Considered "training subjects"

### 2. PRL-labeled subjects
- PRL lesion indices/counts
- Not guaranteed full voxelwise labels
- Used for classification and exploratory analysis
- Also considered "training subjects"

### 3. Count-only subjects
- Only subject-level PRL counts
- No lesion-level mapping
- Used for inference and count comparison only

---

## Image-space assumptions

For the current internal dataset, all relevant files are already aligned.

```
flair.nii.gz
phase.nii.gz
t1.nii.gz
space-flair_seg-lst.nii.gz
lstai_lesion_index.nii.gz
prl_mask_def_prob_*.nii.gz
```

Important:

- No registration or resampling should be performed in the initial implementation
- Preserve affine/header from FLAIR
- Only perform lightweight validation:
  - shape consistency
  - optional affine warnings

Future extension may include registration for external datasets.

---

## LST-AI Lesion Handling

- Use existing lesion masks if available
- Generate indexed lesion mask:
  ```
  0 = background
  1..N = lesion indices
  ```

Missing LST-AI:
- Error for training subjects
- Report for inference-only subjects

---

## ROI Cropping

For each lesion:

1. Extract bounding box
2. Apply padding:
   - `expand_xy`
   - `expand_z`
3. Crop:
   - MRI sequences
   - lesion mask
   - PRL label (if available)

Padding must be configurable.

---

## Image Stacks

Multi-channel inputs from sequences:

- flair
- phase
- t1 (optional)

Stack ordering must be deterministic.

---

## Labels

```
0 = background
1 = lesion
2 = rim
```

Rules:

- PRL lesions: lesion + rim
- Non-PRL lesions: lesion only

---

## Dataset Splitting

- K-fold CV
- Optional held-out test set
- Stratify PRLs
- Splits must be reproducible and independent of preprocessing

---

## Deep Learning Layer

Default:
- MONAI Auto3DSeg SegResNet

Requirements:

- Save generated configs
- Allow extension to other Auto3DSeg models
- Keep hyperparameter flow transparent

---

## Hyperparameter Tuning

Support grid search across:

- preprocessing:
  - expand_xy
  - expand_z
  - image sequences
- training:
  - learning rate
  - epochs
  - loss weights (DiceCE)
  - any Auto3DSeg parameter

Avoid recomputing identical preprocessing outputs.

---

## Inference

Steps:

1. Load preprocessing config from trained run
2. Generate crops
3. Stack images
4. Run model
5. Save:
   - ROI predictions
   - full-brain prediction

---

## Metrics

For labeled subjects:

- Dice (lesion and rim)
- confusion metrics
- sensitivity, specificity, F1

Support:
- per-lesion
- per-subject
- per-run aggregation

---

## Feature Extraction

Extract lesion-level features:

- volumes
- shape (PCA)
- radial metrics
- intensity features

Output:
- one row per lesion

---

## Classification

Support models:

- logistic regression
- SVM
- random forest
- gradient boosting

Capabilities:

- feature selection
- probability outputs
- thresholding
- model saving/loading

---

## Final PRL Outputs

Inputs:
- classifier outputs
- lesion predictions

Output:

```
prl_final_{run_id}.nii.gz
```

Only classifier-approved lesions retained.

---

## Data Provenance

Track:

- preprocessing params
- model config
- run ID
- classifier model

Use manifests rather than relying on directory names.

---

## Pipeline Stages

1. Stage data
2. Generate lesion index
3. Crop ROIs
4. Build dataset manifest
5. Train model
6. Run inference
7. Compute metrics
8. Extract features
9. Train/apply classifier
10. Save final outputs

---

## CLI + Notebook Interface

CLI via Click.

Core logic must be importable for notebook use.

---

## Libraries

Preferred:

- click
- pandas
- numpy
- nibabel
- MONAI
- sklearn

Config options:

- dataclasses
- attrs
- pydantic

---

## Design Constraints

- reproducibility
- idempotency
- scalability (network storage)
- modularity
- extensibility

---

## Implementation Order

1. Config + resources
2. Data staging
3. Lesion indexing
4. Cropping
5. Dataset manifests
6. Training
7. Inference
8. Metrics
9. Features
10. Classification
11. Final outputs

---

## Expected Deliverables

- CLI
- Python API
- dataset/run manifests
- trained models
- metrics outputs
- classifier outputs
- final PRL segmentations

---

## Final Instruction for Implementation Agent

Design architecture before coding:

- directory structure
- config schema
- CLI
- manifests

Then implement incrementally.
