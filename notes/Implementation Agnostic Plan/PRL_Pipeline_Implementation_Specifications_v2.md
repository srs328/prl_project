# PRL Segmentation and Classification Pipeline Specification

This document is intended for a fresh implementation agent. It describes the desired pipeline behavior, data structures, filesystem expectations, labels, resource files, and design requirements **without assuming any existing implementation**.

The implementation should be research-oriented but maintainable. It should support command-line execution and notebook-based exploration.

---

## 1. Goal

Build a modular pipeline for detecting and quantifying **paramagnetic rim lesions / perilesional rim lesions (PRLs)** from MS MRI data.

The pipeline should support:

1. Staging source MRI, LST-AI, and manual PRL segmentation files from an original data root.
2. Reusing existing LST-AI lesion segmentations and lesion-index images.
3. Creating lesion-centered cropped ROIs with tunable padding.
4. Preparing multi-channel MRI image stacks.
5. Training a deep learning segmentation model, defaulting to MONAI Auto3DSeg SegResNet.
6. Running inference on labeled and unlabeled subjects.
7. Computing segmentation metrics when voxelwise ground truth exists.
8. Extracting radiomic, geometric, and image-derived features from candidate rim segmentations.
9. Training and applying lesion-level classifiers.
10. Saving final classifier-filtered full-brain PRL segmentations.
11. Supporting hyperparameter grid searches.
12. Remaining usable from both a CLI and notebooks.
13. Convenient interface for viewing images, segmentations, and inference labels in ITK-SNAP and/or FSLEYES 

The implementation should not be a collection of one-off scripts. Core logic should live in importable modules; CLI commands should be thin wrappers around those modules.

---

## 2. Stable identifiers

The stable subject identifier is:

```text
(subid, sesid)
```

The stable lesion identifier is:

```text
(subid, sesid, lesion_index)
```

`lesion_index` is meaningful only relative to a specific subject/session and the corresponding `lstai_lesion_index.nii.gz`.

Every manifest, metrics table, feature table, classifier output, and final segmentation output should preserve:

```text
subid
sesid
subject_session
lesion_index
```

A convenient working subject-session string is:

```text
sub{subid}-{sesid}
```

For example:

```text
sub1010-20180208
```

This working subject string is a useful convention, but the pipeline should still preserve the explicit `subid` and `sesid` fields.

---

## 3. Source data root and subject/session mapping

### Original data root

Original subject/session data live under:

```bash
ORIG_DATAROOT=/media/smbshare/3Tpioneer_bids
```

Each subject/session is structured like:

```bash
subject_root=${ORIG_DATAROOT}/sub-ms${subid}/ses-${sesid}
```

Example:

```bash
/media/smbshare/3Tpioneer_bids/sub-ms1010/ses-20180208
```

### Subject/session source of truth

The subject/session pairs to use are listed in:

```text
subject-sessions.csv
```

This CSV should be treated as the source of truth for which session to use for each subject.

Expected columns:

```text
sub
ses
```

Interpretation:

- `sub`: numeric subject ID, without the `sub-ms` prefix
- `ses`: session ID/date used in the original data path

Normalize these to:

```text
subid
sesid
```

for all internal manifests.

### Source paths versus working paths

The source path convention is fixed:

```text
${ORIG_DATAROOT}/sub-ms${subid}/ses-${sesid}
```

The implementation may choose its own working data layout. However, every staged subject must retain a mapping back to:

```text
subid
sesid
original_subject_root
working_subject_root
```

The implementation may decide whether to stage data by copying, symlinking, hard-linking, or referencing in-place, but it must preserve provenance.

---

## 4. Subject categories

The pipeline must distinguish three subject/data categories.

These categories are not interchangeable.

### 4.1 Full ground-truth subjects

Subject IDs are listed in:

```text
subjects_with_groundtruth.txt
```

These subjects have:

- MRI files
- LST-AI lesion segmentation files
- indexed lesion files
- manual voxelwise PRL rim segmentations

They can be used for:

- segmentation model training
- validation/test splits
- voxelwise segmentation metrics
- lesion-level classifier training
- end-to-end pipeline testing

These are the only subjects that should be used for voxelwise segmentation training/evaluation by default.

### 4.2 PRL-label subjects

Subject IDs are listed in:

```text
subjects_with_labeled_prls.txt
```

These subjects have lesion-level PRL labels or PRL indices/counts, but they should not automatically be mixed into the main voxelwise segmentation training set unless the implementation verifies that complete voxelwise labels are present and valid.

They may be useful for:

- lesion-level PRL count/reference metadata
- classifier development
- partial validation
- future expansion of the training set

The implementation should keep this distinction explicit.

### 4.3 Count-only / unlabeled inference subjects

Subject IDs are listed in:

```text
subjects_with_counts_only.txt
```

These subjects may have:

- MRI files
- LST-AI lesion segmentation files
- subject-level PRL counts

They do **not** have reliable voxelwise PRL segmentation labels or lesion-index PRL mappings.

They can be used for:

- running inference
- counting predicted PRLs
- comparing predicted counts against available count-only metadata
- downstream exploratory analysis

They cannot be used for voxelwise segmentation metrics unless ground truth is later added.

If count-only rows contain lesion index columns, those lesion indices should be treated as stale/outdated mistakes and ignored.

---

## 5. Supplementary resource files

The implementation may assume the following resource files are provided.

### 5.1 `subject-sessions.csv`

Purpose: maps each subject to the session that should be used.

Expected columns:

```text
sub
ses
```

Use this to construct source paths:

```text
${ORIG_DATAROOT}/sub-ms${sub}/ses-${ses}
```

and internal subject-session IDs:

```text
sub{sub}-{ses}
```

### 5.2 `subjects_with_groundtruth.txt`

One numeric subject ID per line.

These subjects have full voxelwise PRL ground truth and can be used for segmentation training/evaluation.

### 5.3 `subjects_with_labeled_prls.txt`

One numeric subject ID per line.

These subjects have PRL label/count/index information, but are not automatically full voxelwise training subjects.

### 5.4 `subjects_with_counts_only.txt`

One numeric subject ID per line.

These subjects have count-only PRL metadata. Lesion indices should be ignored for these subjects even if accidentally present elsewhere.

### 5.5 `PRL_spreadsheet-lstai_update_label_reference.csv`

Purpose: lesion-index PRL label reference for category 1 and category 2 subjects.

Subjects in categories 1 and 2 have records here.

Important columns:

```text
subid
date_mri
Total PRL
PRL1_label ... PRL20_label
confidence.0 ... confidence.19
```

Interpretation:

- `subid`: numeric subject ID
- `date_mri`: session ID; normalize this to `sesid`
- `PRL<i>_label`: lesion index in `lstai_lesion_index.nii.gz`
- `confidence.<i-1>`: confidence for `PRL<i>_label`
- `Total PRL`: total number of PRLs with `definite` or `probable` ratings

Confidence values:

```text
definite
probable
possible
```

Default behavior:

- Treat `definite` and `probable` as positive PRLs.
- Exclude `possible` by default.
- Make handling of `possible` configurable in the future.

For example, `PRL3_label` corresponds to `confidence.2`.

### 5.6 `PRL_labels_master_full.csv`

Purpose: master PRL metadata table for all subject categories.

This file contains rows for:

- full ground-truth subjects
- PRL-label subjects
- count-only subjects

For category 1 and category 2 subjects, rows should be identical or compatible with `PRL_spreadsheet-lstai_update_label_reference.csv`, with additional columns for counts by PRL category.

Important notes:

- The `PRL` column is a duplicate/summary count of definite + probable PRLs.
- For count-only subjects, `PRL` is the useful count column.
- For count-only subjects, `Total PRL` may be empty and should not be interpreted as zero.
- For count-only subjects, lesion-index columns should be missing. If they exist, treat them as stale/outdated and ignore them.

### 5.7 `groundtruth_labels_to_use.csv`

Purpose: selects which rater suffix/manual segmentation should be used for voxelwise training labels for each full ground-truth subject.

Expected columns:

```text
subid
suffix
```

Example:

```csv
subid,suffix
1010,CH
1011,SRS_CH
1033,LR
1038,CH
```

Behavior:

- For each full ground-truth subject, use the specified suffix to select the manual PRL segmentation used for training.
- Copy or stage all available manual PRL segmentation files for recordkeeping/interrater reliability.
- But use only the selected suffix for primary training label generation.
- If the requested suffix file is missing, fail loudly for that subject and record it in a validation report.
- Do not silently choose another rater file.

---

## 6. Relevant source files

For each subject/session, relevant files may include:

```bash
${subject_root}/flair.nii.gz
${subject_root}/t1.nii.gz
${subject_root}/phase.nii.gz
${subject_root}/lst-ai/space-flair_seg-lst.nii.gz
${subject_root}/lst-ai/lstai_lesion_index.nii.gz
${subject_root}/lst-ai/lesion_pmap.nii.gz
${subject_root}/lesion.t3m20/prl_mask_def_prob_*.nii.gz
```

### 6.1 Required MRI files

Required initially:

```text
flair.nii.gz
phase.nii.gz
```

Optional/future:

```text
t1.nii.gz
```

The initial/default image stack should use:

```text
flair + phase
```

but the selected sequence list must be configurable.

### 6.2 LST-AI files

Primary lesion mask:

```text
lst-ai/space-flair_seg-lst.nii.gz
```

Indexed lesion image:

```text
lst-ai/lstai_lesion_index.nii.gz
```

Optional lesion probability map:

```text
lst-ai/lesion_pmap.nii.gz
```

`lesion_pmap.nii.gz` is not currently incorporated into the main workflow. If it is missing but the binary lesion mask and lesion-index image exist, copying/linking it may fail silently or log a warning.

### 6.3 Manual PRL files

Manual PRL segmentations are expected at:

```text
lesion.t3m20/prl_mask_def_prob_<suffix>.nii.gz
```

where `<suffix>` corresponds to rater initials or reconciliation labels, such as:

```text
CH
LR
SRS
SRS_CH
```

All available manual PRL files should be staged/copied for recordkeeping and interrater reliability work. For primary segmentation training, use only the suffix specified in `groundtruth_labels_to_use.csv`.

---

## 7. Image-space assumptions

For the current internal dataset, the relevant images have already been preprocessed so that files within a subject/session are in the same voxel space, orientation, and shape.

This includes:

```text
flair.nii.gz
phase.nii.gz
t1.nii.gz
lst-ai/space-flair_seg-lst.nii.gz
lst-ai/lstai_lesion_index.nii.gz
lesion.t3m20/prl_mask_def_prob_*.nii.gz
```

Although the original phase image acquisition may have had coarser through-plane resolution, the available `phase.nii.gz` files have already had their headers/spatial metadata fixed to match the FLAIR/T1 space.

For the current project data:

- Assume images are already aligned.
- Do not perform registration or resampling by default.
- Use `flair.nii.gz` as the reference image for affine/header preservation.
- Crops, predictions, and uncropped outputs should preserve the FLAIR reference affine/header where applicable.

The implementation may include lightweight validation, but this should not become a major source of complexity in the first version.

Reasonable lightweight checks:

- verify selected input images have the same array shape
- optionally warn if affines or voxel sizes differ
- record mismatches in a validation report

If these checks fail for the current internal dataset, default behavior can be to raise a clear error or mark the subject as failed. Do not attempt automatic correction in the first implementation.

Advanced image-space handling such as registration, resampling, orientation normalization, or affine reconciliation is a future extension for outside scans.

---

## 8. LST-AI lesion handling

The pipeline starts from LST-AI lesion masks.

The binary lesion mask should be used to generate or validate an indexed lesion image in which each connected lesion has a unique integer ID:

```text
0 = background
1, 2, 3, ... = individual lesion IDs
```

The indexed lesion image is used to:

- define lesion-centered crops
- link spreadsheet PRL lesion indices to image regions
- map predictions back to individual lesions
- generate per-lesion features and classifier rows

### 8.1 Reuse versus generation

The implementation should support:

1. Reusing existing LST-AI outputs if present.
2. Generating `lstai_lesion_index.nii.gz` from `space-flair_seg-lst.nii.gz` if needed.
3. Optionally running LST-AI itself if outputs are missing, as a future extension.

At minimum, missing LST-AI outputs should be logged to a machine-readable report.

### 8.2 Missing LST-AI behavior by subject category

For full ground-truth and PRL-label subjects:

- `space-flair_seg-lst.nii.gz` should exist.
- `lstai_lesion_index.nii.gz` should exist or be generated reproducibly.
- If required LST-AI files are missing and cannot be generated, raise an error.

For count-only subjects:

- If LST-AI files are missing, record the subject in a missing-LST-AI report.
- Do not fail the entire pipeline.

### 8.3 Suggested indexing method

Connected component analysis is probably sufficient for the first implementation.

`c3d` is acceptable, because it is packaged with ITK-SNAP and already familiar in the current workflow.

A Python/scipy/skimage connected-component implementation is also acceptable if it is documented and produces stable lesion indices.

Important: lesion indices must be stable for the PRL spreadsheet mapping to remain valid. If a new lesion index image is generated differently from the original, the spreadsheet labels may no longer match.

---

## 9. Manual PRL segmentation semantics

Manual PRL masks use the following label values:

```text
0 = background
1 = lesion
2 = PRL rim
3 = central vein
4 = extraneous iron content
5 = extralesional hyperintensity
```

For the current segmentation training workflow:

- Use only label `2` from manual PRL masks as the rim.
- Ignore labels `3`, `4`, and `5`.
- Do not use manual label `1` as the lesion core for training.
- Derive the lesion core from LST-AI instead.

Derived training labels should use:

```text
0 = background
1 = LST-AI lesion body
2 = manual PRL rim
```

### 9.1 Derived rim mask

For selected suffix `<suffix>`, extract label `2` from:

```text
prl_mask_def_prob_<suffix>.nii.gz
```

and save a derived rim-only mask such as:

```text
prl_rim_def_prob_<suffix>.nii.gz
```

### 9.2 Derived training label

Overlay the rim-only mask onto the LST-AI lesion mask to create the actual training segmentation.

The training label convention should be:

```text
0 = background
1 = lesion body from LST-AI
2 = PRL rim from manual segmentation
```

The implementation should preserve provenance linking every derived training label to:

```text
subid
sesid
lesion_index
selected suffix
source manual PRL file
source LST-AI lesion mask
derived rim file
derived training label file
```

### 9.3 Future label extensions

Central vein sign and other label classes may be incorporated later, but they are out of scope for the initial implementation.

---

## 10. Lesion-level training case construction

For full ground-truth subjects:

1. Use `lstai_lesion_index.nii.gz` to enumerate lesion indices.
2. Use `PRL_spreadsheet-lstai_update_label_reference.csv` to identify which lesion indices are PRLs.
3. Treat lesions with confidence in `["definite", "probable"]` as positive PRL lesions by default.
4. Exclude `possible` PRLs by default.
5. Treat all other indexed lesions as non-PRL lesions by default, unless explicitly excluded.
6. Use the selected manual rater suffix from `groundtruth_labels_to_use.csv` to derive voxelwise rim labels for positive PRL lesions.

For non-PRL lesions, the training label should contain lesion body but no rim.

For PRL lesions, the training label should contain lesion body and rim.

Open/default behavior for `possible` PRLs:

- Recommended default: exclude from segmentation training and classifier training.
- Alternative modes may be supported:
  - treat as positive
  - treat as negative
  - keep as separate uncertain class

The chosen behavior must be recorded in dataset metadata.

---

## 11. Data staging

The implementation may choose whether to copy, symlink, hard-link, or reference original files, but it must produce a clear staged-data manifest.

### 11.1 Inputs

```text
ORIG_DATAROOT
subject-sessions.csv
subjects_with_groundtruth.txt
subjects_with_labeled_prls.txt
subjects_with_counts_only.txt
PRL_spreadsheet-lstai_update_label_reference.csv
PRL_labels_master_full.csv
groundtruth_labels_to_use.csv
```

### 11.2 Required staging behavior

For each subject/session:

1. Resolve `subject_root` from `ORIG_DATAROOT`, `subid`, and `sesid`.
2. Stage MRI files.
3. Stage LST-AI files.
4. Stage all available manual PRL files, when present.
5. Record subject category.
6. Record missing files.
7. Preserve source-to-working path mapping.

### 11.3 Staging manifest

A staging manifest should contain at least:

```text
subid
sesid
subject_session
category
original_subject_root
working_subject_root
flair_path
phase_path
t1_path
lst_mask_path
lst_index_path
lesion_pmap_path
manual_prl_files
selected_groundtruth_suffix
selected_groundtruth_file
missing_required_files
missing_optional_files
status
```

---

## 12. ROI cropping and padding

Each lesion should be cropped into an individual ROI centered around the corresponding indexed lesion.

For each lesion:

1. Find the tight bounding box around the lesion in `lstai_lesion_index.nii.gz`.
2. Expand the bounding box by tunable padding:
   - `expand_xy`: padding in x/y plane
   - `expand_z`: padding in z direction
3. Crop each selected MRI sequence using the expanded bounding box.
4. Crop the LST-AI lesion mask.
5. For training subjects, crop the selected derived training label.
6. Save or record crop metadata.

The implementation must handle bounding boxes that extend beyond image boundaries. It should either mimic `fslroi` behavior with zero padding or implement a clearly documented equivalent.

Padding parameters are experimental hyperparameters and must not be hard-coded.

Example tunable parameters:

```yaml
preprocessing:
  expand_xy: [10, 20, 30]
  expand_z: [1, 2, 3]
```

Because `expand_xy`, `expand_z`, and image-stack composition are tunable, the same lesion may have multiple derived crop files corresponding to different experiment configurations. File naming, metadata, and caching must account for this.

### 12.1 Cropping implementation reference

The following shell logic was provided as a useful starting point. The implementation can adapt it or replace it with an equivalent Python implementation.

```bash
n_lesions=$(fslstats lstai_lesion_index.nii.gz -R | awk '{printf "%d\n", $2}')
echo "Found ${n_lesions} lesions"
> "$bounding_boxes"
for lesion_label in $(seq 1 "$n_lesions"); do
    temp_BBox=$(fslstats lstai_lesion_index.nii.gz \
        -l $(bc <<< "${lesion_label} - 0.5") \
        -u $(bc <<< "${lesion_label} + 0.5") \
        -w)
    echo "$lesion_label $temp_BBox" >> "$bounding_boxes"
done

echo "will expand xy dimension of PRL by $expand_xy"
expanded_boxes="lstai_bounding_boxes_xy${expand_xy}_z${expand_z}.txt"
while read -r lesion_label bbox_rest; do
    roi_boundaries=$(echo "$bbox_rest" | awk -v expand_xy="$expand_xy" -v expand_z="$expand_z" '{
        printf "%d %d %d %d %d %d\n", $1-expand_xy, $2+2*expand_xy, $3-expand_xy, $4+2*expand_xy, $5-expand_z, $6+2*expand_z
    }')
    echo "$lesion_label $roi_boundaries" >> "$expanded_boxes"
done < "$bounding_boxes"

echo "ROI bounding boxes saved to ${expanded_boxes}"
```

### 12.2 Crop metadata

For each crop, record:

```text
subid
sesid
subject_session
lesion_index
bbox_original
bbox_expanded
expand_xy
expand_z
image_channels
crop_paths
label_crop_path
source_subject_root
```

---

## 13. MRI image stacks

The model input is a multi-channel image stack created from selected MRI sequences.

Possible sequences:

```text
flair
phase
t1
```

Initial/default stack:

```text
flair + phase
```

The sequence list must be tunable.

Example:

```yaml
preprocessing:
  images:
    - [flair, phase]
    - [flair, phase, t1]
```

Requirements:

- Stack ordering must be deterministic.
- Stack ordering must be recorded in metadata.
- The same ordering must be used for training and inference.
- Generated stack files should encode image list and crop padding in their filename or manifest metadata.

---

## 14. Dataset splitting

For subjects with full ground truth, the pipeline should create reproducible training/validation/test splits.

Requirements:

- Support k-fold cross-validation.
- Support optional held-out test split.
- Preserve fold assignments across experiments.
- Stratify or balance PRL cases across folds as much as practical.
- Store split assignments in a durable dataset manifest.
- Split manifest should be independent of preprocessing hyperparameters.

Cropping and image-stack parameters should not change which lesion belongs to which fold.

### 14.1 Dataset manifest

A dataset manifest should include one row per lesion/case and at least:

```text
subid
sesid
subject_session
lesion_index
category
case_type
confidence
split
fold
is_prl
is_possible_prl
selected_suffix
image_stack_key
label_path
lesion_mask_path
crop_dir_or_paths
```

`case_type` can be something like:

```text
PRL
Lesion
PossiblePRL
CountOnly
InferenceOnly
```

The exact values can differ, but they should be documented.

---

## 15. Deep learning segmentation layer

### 15.1 Default model

Default deep learning model:

```text
MONAI Auto3DSeg SegResNet
```

The implementation should use [MONAI Auto3DSeg](https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/README.md) in a way that preserves access to:

- generated configs
- fold outputs
- training logs
- model artifacts
- final hyperparameters

### 15.2 Extensibility

Although SegResNet is the default, the implementation should be extensible to other Auto3DSeg-supported algorithms.

Avoid hard-coding assumptions such as:

```text
segresnet_0/
segresnet_1/
...
```

except where encapsulated behind an algorithm-specific interface.

Model-specific behavior should be isolated behind a config or adapter layer.

### 15.3 Auto3DSeg hyperparameters

The implementation should research or document how Auto3DSeg hyperparameters flow into generated `hyper_parameters.yaml` files.

Desired behavior:

- User-facing experiment configs define training hyperparameters.
- Hyperparameters flow predictably into Auto3DSeg.
- Generated Auto3DSeg configs are saved for provenance.
- Completed runs expose the actual hyperparameters used.
- Nested key syntax or template-specific parameter names are translated in one place and tested.

---

## 16. Hyperparameter tuning

The pipeline should support grid-style hyperparameter experiments.

Important tunable preprocessing parameters:

```yaml
expand_xy
expand_z
images
```

Important tunable training parameters:

```yaml
learning_rate
num_epochs
batch_size / num_images_per_batch
num_crops_per_image
roi_size
```

Important loss-related parameters:

```yaml
DiceCE weights
DiceCE lambda_dice
DiceCE lambda_ce
class weights
```

The implementation should be flexible enough to tune any parameter that Auto3DSeg exposes through `hyper_parameters.yaml`, not just the parameters listed above.

A grid config should support Cartesian products of preprocessing and training parameters.

Example:

```yaml
dataset: roi_train2
experiment_name: dicece_weight_sweep

preprocessing:
  expand_xy: [20]
  expand_z: [2]
  images:
    - [flair, phase]

training:
  learning_rate: [0.0001, 0.0002]
  num_epochs: [250]
  loss:
    _target_: DiceCELoss
    lambda_dice: [0.5, 1.0]
    lambda_ce: [0.5, 1.0]
    weight:
      - [1.0, 1.0, 1.0]
      - [1.0, 1.0, 15.0]
```

The exact config syntax can differ, but this level of flexibility should be preserved.

The pipeline should avoid regenerating identical crops/image stacks when only training hyperparameters change.

---

## 17. Inference

The pipeline should support inference on:

1. held-out labeled test cases
2. cross-validation folds
3. fresh unlabeled subjects
4. count-only subjects

For fresh subjects:

1. Load the trained run’s preprocessing configuration.
2. Ensure LST-AI lesion masks and indexed lesion masks exist.
3. Generate lesion-centered crops using the same padding as training.
4. Stack the same MRI sequences in the same order as training.
5. Run the trained segmentation model.
6. Save per-lesion ROI predictions.
7. Uncrop predictions back into full-brain subject space.

Outputs should include:

```text
per-lesion ROI prediction files
full-brain prediction file
machine-readable inference manifest
failure report for subjects or lesions that could not be processed
```

The full-brain prediction should preserve the label convention:

```text
0 = background
1 = lesion body
2 = PRL rim
```

### 17.1 Inference manifest

Recommended columns:

```text
subid
sesid
subject_session
lesion_index
run_id
model_algorithm
preprocessing_config
image_stack_key
roi_prediction_path
full_brain_prediction_path
status
error_message
```

---

## 18. Evaluation metrics

For subjects with voxelwise ground truth, compute segmentation metrics.

At minimum:

- lesion Dice
- rim / PRL Dice
- TP, FP, TN, FN for rim voxels
- sensitivity / recall
- specificity
- precision
- negative predictive value
- accuracy
- F1

Optional:

- Hausdorff distance
- 95th percentile Hausdorff distance
- per-lesion PRL detection accuracy
- subject-level PRL count error

Metrics should be saved at multiple levels:

1. per-lesion
2. per-subject
3. per-run aggregate
4. cross-run / grid summary

The implementation should clearly identify whether a metric is voxelwise, lesionwise, or subjectwise.

---

## 19. Radiomic / geometric feature extraction

The pipeline should include a feature extraction layer for candidate PRL/rim predictions.

Candidate lesions should generally be those with inferred rim voxels near or overlapping the indexed lesion of interest.

Features may include:

- rim voxel count
- rim volume
- lesion volume
- rim convex hull volume
- lesion convex hull volume
- enclosing sphere radius
- PCA-derived shape features
- radial distribution features
- intensity features from phase, FLAIR, T1, or other sequences
- pyradiomics features, if useful

The design should allow features to be added without rewriting the classifier workflow.

Feature extraction should produce one row per subject-lesion-model prediction.

Recommended columns:

```text
subid
sesid
subject_session
lesion_index
dataset_name
run_id
inference_path
has_candidate_rim
feature_1
feature_2
...
label_if_available
```

---

## 20. Lesion-level classification layer

The pipeline should include a lesion-level classification layer on top of segmentation.

Purpose:

- The deep learning model may generate candidate rim segmentations.
- A classifier decides whether each candidate rim should count as a true PRL.

The classification layer should make it convenient to compare:

- logistic regression
- support vector machine / SVM
- random forest
- gradient boosting
- XGBoost or LightGBM, if installed
- simple threshold rules
- calibrated classifiers

Design requirements:

- selecting feature columns
- training/test splits
- cross-validation
- class weighting
- imputation
- scaling
- probability outputs
- threshold tuning
- saving fitted models
- applying saved models to new inference datasets

Classifier outputs should include:

```text
subid
sesid
subject_session
lesion_index
run_id
has_candidate_rim
predicted_prl
prl_probability
classifier_name
classifier_version_or_model_path
feature_set_name
```

The classifier should be usable both from CLI and notebooks.

---

## 21. Final classifier-filtered PRL segmentations

After classification, save final full-brain PRL outputs.

Inputs:

- per-lesion inference predictions
- classifier output table
- subject-level image metadata
- lesion bounding boxes
- selected list of accepted lesion indices

Output:

```text
full-brain classifier-filtered PRL segmentation
```

The final segmentation should include only classifier-approved PRL predictions.

Open design choice:

Either preserve full segmentation labels:

```text
0 = background
1 = lesion body
2 = PRL rim
```

or save a rim-only PRL mask:

```text
0 = background
1 = final PRL rim
```

The implementation should make this explicit and ideally support both.

Recommended outputs per subject:

```text
prl_candidates_<run_id>.nii.gz
prl_final_<run_id>_<classifier_name>.nii.gz
prl_final_<run_id>_<classifier_name>.csv
```

The final CSV should record exactly which lesion indices were retained.

---

## 22. Data management and provenance

Every major derived artifact should be traceable to its inputs and parameters.

The pipeline should save metadata for:

- source data root
- subject/session file used
- subject category
- selected rater suffix
- preprocessing parameters
- image stack
- split/fold assignment
- model algorithm
- Auto3DSeg config
- training hyperparameters
- run ID
- inference model/run used
- classifier model and feature set used

Avoid relying on folder names alone for provenance.

Use machine-readable manifests where possible.

---

## 23. Suggested pipeline stages

The implementation does not need to use these exact names, but it should support equivalent stages.

### 23.1 Stage source data

Inputs:

```text
ORIG_DATAROOT
subject-sessions.csv
manual PRL label resources
```

Outputs:

```text
working data root
staging manifest
missing file reports
```

Responsibilities:

- copy/symlink/link/reference MRI files
- copy/symlink/link/reference LST-AI outputs
- stage manual labels
- record missing files
- preserve subject/session identity

### 23.2 Prepare lesion masks

Input:

```text
space-flair_seg-lst.nii.gz
```

Outputs:

```text
lstai_lesion_index.nii.gz
lesion metadata table
```

Responsibilities:

- connected-component labeling
- lesion indexing
- lesion count reporting

### 23.3 Prepare crops

Inputs:

```text
indexed lesion mask
MRI images
manual labels if available
preprocessing config
```

Outputs:

```text
per-lesion image crops
per-lesion label crops
bounding box files or manifest
```

Responsibilities:

- bounding box generation
- padding
- boundary handling
- image/label crop generation

### 23.4 Build dataset manifest

Inputs:

```text
subject metadata
lesion metadata
label availability
selected rater suffixes
split config
```

Outputs:

```text
dataset manifest
fold assignment manifest
training datalist
```

Responsibilities:

- PRL vs non-PRL labeling
- possible-PRL handling
- fold/test split assignment
- paths to images and labels

### 23.5 Train segmentation model

Inputs:

```text
training datalist
Auto3DSeg config
training hyperparameters
```

Outputs:

```text
trained model folders
saved Auto3DSeg configs
training logs
run metadata
```

### 23.6 Run segmentation inference

Inputs:

```text
trained model run
subject data
preprocessing config
```

Outputs:

```text
per-lesion ROI predictions
full-brain prediction
inference manifest
```

### 23.7 Evaluate segmentation

Inputs:

```text
predictions
ground truth labels
```

Outputs:

```text
per-lesion metrics
per-subject metrics
per-run metrics
```

### 23.8 Extract features

Inputs:

```text
candidate prediction ROIs
indexed lesion masks
MRI sequences
bounding boxes
```

Output:

```text
radiomic/geometric feature table
```

### 23.9 Train/apply classifier

Inputs:

```text
feature table
labels if available
classifier config
```

Outputs:

```text
fitted classifier
classification report
classified lesion table
```

### 23.10 Save final PRL outputs

Inputs:

```text
classified lesion table
per-lesion predictions
subject image metadata
```

Outputs:

```text
classifier-filtered full-brain PRL segmentation
final subject-level PRL counts
```

---

## 24. CLI and notebook interface

The pipeline should expose a CLI, preferably using Click.

Example CLI shape:

```bash
prl stage-data --config config.yaml
prl prepare-lesions --dataset roi_train2
prl preprocess --dataset roi_train2 --expand-xy 20 --expand-z 2 --images flair --images phase
prl train --dataset roi_train2 --config train.yaml
prl grid --config grid.yaml
prl predict --run-dir /path/to/run
prl infer --run-dir /path/to/run --subjects-file subjects.txt
prl metrics --run-dir /path/to/run
prl extract-features --run-dir /path/to/run --dataset inference_dataset
prl train-classifier --features features.csv --config classifier.yaml
prl apply-classifier --features features.csv --model model.pkl
prl save-final --classification-csv classified.csv --run-dir /path/to/run
```

Core logic should live in importable Python functions/classes, not only CLI functions.

Notebook-friendly usage should look roughly like:

```python
dataset = load_dataset("roi_train2")
run = load_run("/path/to/run")
features = extract_features(run, dataset)
model = train_classifier(features)
save_final_outputs(run, model, dataset)
```

### 24.1 MRI viewing interface

I prefer ITK-SNAP for opening MRI images and segmentations. Sample command to open three image files and three segmentations:

```bash
itksnap -g image1.nii.gz -o image2.nii.gz image3.nii.gz -s seg1.nii.gz seg2.nii.gz seg3.nii.gz
```

Sometimes I use fsleyes for viewing as well:

```bash
fsleyes -xh -yh \
  image1.nii.gz -ot volume \
  image2.nii.gz -ot volume \
  image3.nii.gz -ot volume \
  seg1.nii.gz -ot label \
  seg2.nii.gz -ot label \
  seg3.nii.gz -ot label
```

---

## 25. Library preferences and design freedom

I use a mix of Python and bash scripting. Prefer bash scripting when there are more complex tasks requiring the use of command line neuroimaging processing. Preferred tools:
- `c3d` and `fslmaths` for nifti manipulation. 
  - `c3d` comes packaged with itksnap and it is already on my PATH and ready to use
  - I have FSL but to use it I need to activate it. I call the following function defined in my ~/.bashrc, so an equivalent may be necessary in a project specific environment or sourcing file
    ```bash
    function activate_fsl() {
      # FSL Setup
      FSLDIR=~/fsl
      # PATH=${FSLDIR}/share/fsl/bin:${PATH} # fsl.sh does this
      # export FSLDIR PATH
      export FSLDIR
      . ${FSLDIR}/etc/fslconf/fsl.sh
    }
    ```
- `ANTS` for MRI image registration if that ever becomes necessary, which I don't currently imagine it will 
- I am open to installing other neuroimaging tools if you determine they would be useful

For Python, feel free to look beyond my recommendations below. Consider these a starting point. 

Preferred or acceptable libraries:

- `click` for CLI
- `pandas`
- `numpy`
- `nibabel` for most nifti work
- `nipype` for interfacing with fsl and neuroimaging tools
- `nilearn` for visualization of mri 
- `scipy`
- `scikit-image`
- `scikit-learn`
- `MONAI`
- `pyradiomics`, if useful
- `joblib`, `concurrent.futures`, or multiprocessing for parallelism
- `tqdm`
- `loguru` or standard `logging`
- `attrs`, `dataclasses`, or `pydantic` for configuration objects
- `pyyaml` or `omegaconf` for configs 

Guidance:

- `dataclasses`: good default, minimal dependency, but actually I'd prefer attrs or pydantic
- `attrs`: good for lightweight immutable configs and validation.
- `pydantic`: good for strong validation and serialization.
- `omegaconf` / Hydra: useful for complex ML configs but may add complexity.
- Plain dictionaries are acceptable internally, but avoid letting unvalidated dicts spread throughout the pipeline.

The implementer may choose, but should justify the choice and keep the configuration layer consistent.

---

## 26. Design freedom versus constraints

The implementation may decide:

- copied vs symlinked vs referenced staged data
- one working data root vs separate roots for training/inference/count-only subjects
- whether crops live inside subject folders or in a separate derived-data cache
- exact internal Python class names
- whether configs use dataclasses, attrs, pydantic, or another schema layer
- exact output directory layout

The implementation must preserve:

- stable subject/session identity
- stable lesion identity as `(subid, sesid, lesion_index)`
- traceability to original files
- selected rater suffix for training labels
- preprocessing parameters used for every crop/image stack
- model/run ID for every prediction
- classifier/model ID for every final PRL output

---

## 27. Design considerations

### 27.1 Reproducibility

Every run should be reproducible from saved configs and manifests.

Save:

- config files
- generated datalists
- run metadata
- model hyperparameters
- feature-set definitions
- classifier model metadata

### 27.2 Idempotency

Pipeline stages should be safe to rerun.

Each stage should define behavior for:

- skip existing outputs
- overwrite existing outputs
- regenerate missing outputs only

### 27.3 Scalability

Data may live on network-mounted storage. Avoid unnecessary repeated recursive scans and excessive file existence checks.

Use manifests and cached file inventories when possible.

Preprocessing and inference should support parallel execution over subjects or lesions where safe.

### 27.4 Extensibility

The design should allow future additions:

- new MRI sequences
- new Auto3DSeg algorithms
- new loss functions
- new classifiers
- new radiomic features
- new output conventions
- new subject categories
- future image registration/resampling for external datasets

### 27.5 Separation of concerns

Avoid putting all logic in a single script.

Suggested separation:

```text
data staging
resource parsing
dataset manifests
preprocessing / crops
model training
inference
metrics
feature extraction
classification
final output generation
visualization helpers
```

### 27.6 Human review

Preserve artifacts useful for manual inspection:

- cropped ROIs
- per-lesion predictions
- full-brain predictions
- classifier probabilities
- rejected candidate PRLs
- subject-level summaries

Optional future feature: generate ITK-SNAP workspace files or equivalent visualization helpers.

---

## 28. Important open design questions

The implementer should make explicit choices about:

1. Should staged data be copied, symlinked, hard-linked, or referenced in-place?
2. Should all subjects live under one working data root, or should training/inference/count-only subjects be separated?
3. Should crops be stored inside each subject folder or in a derived-data cache?
4. How should different crop/image-stack parameter combinations be named?
5. How should multiple manual label suffixes per subject be represented?
6. How should classifier-filtered outputs encode lesion body vs rim?
7. Should final outputs be saved in subject folders, run folders, or both?
8. How should failed subjects or failed lesions be recorded?
9. How should grid-search results be summarized across runs?
10. How should `possible` PRLs be handled?

---

## 29. Recommended implementation order

Implement incrementally:

1. Config schemas and resource loading.
2. Source-data scanner/stager with missing-file reports.
3. LST-AI/indexed-lesion validation.
4. Dataset manifest creation.
5. Crop/image-stack generation.
6. Training datalist creation.
7. Auto3DSeg SegResNet training for one dataset.
8. Validation/test inference.
9. Metrics for ground-truth cases.
10. Fresh-subject inference.
11. Feature extraction.
12. Classifier training/application.
13. Final classifier-filtered full-brain output saving.
14. Grid/HPO orchestration.

At each step, write a manifest and make the step rerunnable/idempotent.

---

## 30. Expected deliverables

A good first implementation should provide:

1. A documented config format.
2. A CLI with clear pipeline stages.
3. Importable Python APIs for notebook use.
4. Resource parsers for all CSV/TXT metadata files.
5. Dataset and run manifests.
6. LST-AI reuse / missing-file reporting.
7. Lesion indexing.
8. ROI cropping with tunable padding.
9. Image-stack creation with tunable sequences.
10. Ground-truth label derivation from manual rim labels + LST-AI lesion masks.
11. MONAI Auto3DSeg SegResNet training.
12. Grid-search support.
13. Inference on fresh subjects.
14. Segmentation metrics for labeled cases.
15. Feature extraction for predicted rims.
16. Classifier training/application.
17. Final classifier-filtered full-brain PRL outputs.
18. Basic logging and error reports.

---

## 31. Suggested prompt to give Codex / ChatGPT

```text
I want you to design and implement a fresh PRL segmentation and classification pipeline from this specification. Do not assume any existing implementation. Start by proposing an architecture, module layout, config schema, data manifests, and CLI commands. Then implement the pipeline incrementally.

Prioritize:
1. clear data provenance
2. reproducible configs/manifests
3. modular design
4. notebook-friendly APIs
5. CLI usability
6. extensibility to Auto3DSeg models and classifier algorithms
7. correct parsing of the resource CSV/TXT files and subject filesystem

Before writing code, identify open design decisions and propose defaults.
```
