
# PRL Segmentation and Classification Pipeline Specification

## Goal

Build a research-oriented but maintainable pipeline for detecting and quantifying **paramagnetic rim lesions / perilesional rim lesions (PRLs)** from MS MRI data.

The pipeline should support:

1. Copying / staging source MRI and lesion segmentation files from an original data root.
2. Running or reusing LST-AI lesion segmentations.
3. Creating lesion-centered cropped ROIs with tunable padding.
4. Preparing multi-channel MRI image stacks.
5. Training a deep learning segmentation model, defaulting to MONAI Auto3DSeg SegResNet.
6. Running inference on both labeled and unlabeled subjects.
7. Computing segmentation metrics when ground truth exists.
8. Extracting radiomic / geometric features from candidate rim segmentations.
9. Training and applying a lesion-level classifier.
10. Saving final classifier-filtered full-brain PRL segmentations.
11. Supporting hyperparameter grid searches.
12. Remaining usable both from a CLI and from notebooks.

The implementation should be modular and extensible. It should not be built as a collection of one-off scripts.

---

# Lesion segmentation with LST-AI

The pipeline starts from LST-AI lesion masks.

The LST-AI binary lesion mask should be used to generate an indexed lesion image in which each connected lesion has a unique integer ID:

```text
0 = background
1, 2, 3, ... = individual lesion IDs
```

This indexed lesion image is used to define lesion-centered crops and to map predictions back to individual lesions.

The implementation should support two modes:

1. Reuse existing LST-AI outputs if present (raise error if not present for category 1 and 2 subjects).
2. Optionally run LST-AI if outputs are missing, if the environment supports it.

At minimum, missing LST-AI outputs should be logged to a machine-readable report.

Suggestion for indexing: connected component analysis is probably most feasible at this stage, and I am most familiar with `c3d`, the tool that is packaged with itksnap for this purpose.


---

# Subject categories

The pipeline needs to distinguish at least three subject/data categories.

## 1. Full ground-truth subjects

These subjects have MRI, LST-AI lesion segmentations, and manual PRL rim segmentations. Their subid's are in subjects_with_groundtruth.txt

They can be used for:

* training segmentation models
* validation/test splits
* segmentation metrics
* radiomic/classifier training
* end-to-end pipeline testing

## 2. PRL-label subjects

These subjects have manual PRL labels (will explain how this information is recorded later), possibly usable for lesion-level classification, count comparison, or partial validation, but do not have complete voxelwise ground truth in the same format as the core training set. Their subid's are in subjects_with_labeled_prls.txt

The implementation should keep this distinction explicit rather than quietly mixing these subjects into the main segmentation training set.

## 3. Count-only / unlabeled inference subjects

These subjects may have MRI, LST-AI lesion segmentations, and perhaps spreadsheet-level PRL counts, but no voxelwise PRL segmentation labels. Their subid's are recorded in `subjects_with_counts_only.txt`

They can be used for:

* running inference
* counting predicted PRLs
* comparing predicted counts against available count-only metadata
* downstream exploratory analysis

They cannot be used for voxelwise segmentation metrics unless ground truth is later added.

## Records of ground truth, PRL labels, and PRL counts

### Lesion Indices and Counts

Subjects in categories 1 and 2 have records in `PRL_spreadsheet-lstai_update_label_reference.csv`.
It maps each PRL to its corresponding index in `lstai_lesion_index.nii.gz`. Each subject gets a row ("subid"). The column "date_mri" contains the sesid (so `f"sub{subid}-{date_mri}"` names the `subject_root` folder)
    - The columns `PRL<i>_label` contain the index for `lstai_lesion_index` 
    - The columns `confidence.<i-1>` label the confidence about whether the identified PRL really is a PRL (these were professionally rated by neurologist CH and his neuroradiologist collegue a few years ago for a different project). The three values are `["definite", "probable", "possible"]`. For now I am only using the definite and probable labels.
    - The column `Total PRL` refers to the total number of PRL with definite or probable ratings

The records in `PRL_labels_master_full.csv` document subjects in all three categories. The rows for categories 1 and 2 should be identical to rows in `PRL_spreadsheet-lstai_update_label_reference.csv` but there are additional columns with counts for each PRL category (definite, probably, possible) as well as a duplicate column for total PRL count for definite and probable (column titled PRL). There are rows for category 3 subjects as well. These rows have columns filled in for counts (see column PRL). The column "Total PRL" is empty for these subjects  Lesion indices should be missing for all rows corresponding to subjects listed in `subjects_with_counts_only.txt`. If somehow any of these rows do have an index, it's a mistake (they correspond to an outdated lesion index which has been lost to time).

### Ground truth segmentation

Manual rim segmentations are labeled by rater initials or suffixes. A given subject may have multiple candidate PRL label files from different raters or reconciliation stages. It is important to copy all of them over to the subject folders in the working dataroot because they will be used for interrater reliability. But for deep learning training, I noted which segmentations to use in the file `groundtruth_labels_to_use.csv`. The csv head looks like:

```csv
subid,suffix
1010,CH
1011,SRS_CH
1033,LR
1038,CH
```

Expected behavior:

* Load subject-to-label-suffix mapping from a resource file.
* For each training subject, use the specified rater suffix to find the correct manual PRL segmentation.
* Report missing or ambiguous labels.
* Preserve the selected suffix in dataset/run metadata for reproducibility.

Manual label files should be treated as immutable inputs. Any derived rim-only masks or training labels should be saved as derived artifacts with clear names.

In these manual segmentation masks, rims are labeled 2, lesion is labeled 1. In some cases, there are additional labels (3=central vein, 4=extraneous iron content; 5=extralesional hyperintensity). All but rim (label 2) are ignored. The ground truth values for lesion core are derived from LST-AI, and CVS, extraneous iron, and extralesional hyperintensity are to be ignored. In preprocessing, the rims should extracted from this image and saved into  "prl_rim_def_prob_\<rater\>.nii.gz" and overlayed onto `space-flair_seg-lst.nii.gz` to get the actual training segmentations. Eventually, we may add the central vein sign if rim segmentation is successful.


---

# Data sources

## Original data root

Original subject/session data live under:

```bash
ORIG_DATAROOT=/media/smbshare/3Tpioneer_bids
```

Each subject/session is structured like:

```bash
subject_root=${ORIG_DATAROOT}/sub-ms${subid}/ses-${sesid}
```

The subject/session pairs to use are listed in:

```bash
subject-sessions.csv
```

This CSV should be treated as the source of truth for which session to use for each subject.

## Relevant source files

For each subject/session, the relevant files are:

```bash
${subject_root}/flair.nii.gz
${subject_root}/t1.nii.gz
${subject_root}/phase.nii.gz
${subject_root}/lst-ai/space-flair_seg-lst.nii.gz #lesion mask file
${subject_root}/lst-ai/lstai_lesion_index.nii.gz #lesion indexed with connected component analysis using c3d
${subject_root}/lst-ai/lesion_pmap.nii.gz #lesion probability map; if this does not exist but the other lst-ai files do, that's okay; copying/linking to working directory can fail silently or log a warning since the lesion pmap is not yet incorporated into any workflow
${subject_root}/lesion.t3m20/prl_mask_def_prob_*.nii.gz #segmented rim where the wildcard corresponds to initials of the rater
```

Regarding lst-ai files: if they exists, they should be copied or linked into the pipeline’s working data structure. 

lst-ai files (except lesion_pmap.nii.gz) should definitely exist for any subject who has ground truth segmentations or labeled PRL because the PRL labels correspond to the index in ${subject_root}/lst-ai/lstai_lesion_index.nii.gz. If not, an error should be raised

If these files do not exist for subjects who just have counts only, the subject should be recorded in a missing-LST-AI report. The pipeline should be designed so that LST-AI could be run automatically in the future, but it is acceptable for the first implementation to only detect and report missing LST-AI outputs.

The pipeline should not assume that all subjects have manual PRL annotations.

---

# ROI cropping and padding

Each lesion should be cropped into an individual ROI centered around the corresponding indexed lesion.

For each lesion:

1. Find the tight bounding box around the lesion in the indexed lesion image.
2. Expand the bounding box by tunable padding:

   * `expand_xy`: padding in x/y plane
   * `expand_z`: padding in z direction
3. Crop each selected MRI sequence using the expanded bounding box.
4. Crop the LST-AI lesion mask.
5. For training subjects, crop the selected manual PRL label / derived PRL label.

The implementation must handle bounding boxes that extend beyond image boundaries. It should either mimic `fslroi` behavior with zero padding or implement a clearly documented equivalent.

Padding parameters are experimental hyperparameters and must not be hard-coded.

Example tunable parameters:

```yaml
preprocessing:
  expand_xy: [10, 20, 30]
  expand_z: [1, 2, 3]
```

Design consideration: because `expand_xy`, `expand_z`, and image-stack composition are tunable, the same lesion may have multiple derived crop files corresponding to different experiment configurations. File naming, metadata, and caching should account for this.

The following snippet was provided by CH, consider adapting this for the cropping and paddng:

```bash
n_lesions=$(fslstats lstai_lesion_index.nii.gz -R | awk '{printf "%d\n", $2}')
echo "Found ${n_lesions} lesions"
> "$bounding_boxes"   # create empty
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

---

# MRI image stacks

The model input is a multi-channel image stack created from selected MRI sequences.

Possible sequences include:

```text
flair
phase
t1
```

The initial/default stack should likely be:

```text
flair + phase
```

but the sequence list must be tunable.

Example:

```yaml
preprocessing:
  images:
    - [flair, phase]
    - [flair, phase, t1]
```

The pipeline should make image-stack ordering deterministic, record it in metadata, and use the same ordering for training and inference.

---

# Label convention

Training labels should use a consistent segmentation convention.

Recommended convention:

```text
0 = background
1 = lesion body
2 = PRL rim
```

For non-PRL lesions, the label should contain the lesion body but no rim.

For PRL lesions, the label should contain both lesion body and rim.

The exact rule for deriving the rim from manual labels should be explicit and reproducible. For example, if manual PRL labels contain a rim class, extract the rim class and combine it with the LST-AI lesion mask to create the final training label.

The pipeline should preserve enough metadata to trace each training label back to:

* subject
* lesion index
* rater suffix
* source manual label file
* crop padding
* image stack

---

# Dataset splitting

For subjects with full ground truth, the pipeline should create reproducible training/validation/test splits.

Design requirements:

* Support k-fold cross-validation.
* Hold out a test split if desired.
* Preserve fold assignments across experiments so hyperparameter sweeps are comparable.
* Stratify or balance PRL cases across folds as much as practical.
* Store split assignments in a durable dataset manifest.

The split manifest should be independent of preprocessing hyperparameters. Cropping and image-stack parameters should not change which lesion belongs to which fold.

---

# Deep learning segmentation layer

## Default model

The default deep learning layer should use:

```text
MONAI Auto3DSeg SegResNet
```

The implementation should use MONAI Auto3DSeg in a way that preserves access to generated configs and trained fold outputs.

## Extensibility

Although SegResNet is the default, the implementation should be extensible to other Auto3DSeg-supported algorithms.

The design should avoid hard-coding assumptions like:

```text
segresnet_0/
segresnet_1/
...
```

except where this is unavoidable or clearly encapsulated.

Model-specific behavior should be isolated behind a small interface or config layer.

## Auto3DSeg hyperparameters

The implementation should research or document how Auto3DSeg hyperparameters flow into the generated `hyper_parameters.yaml` files.

Desired behavior:

* A user-facing experiment config should define training hyperparameters.
* Those hyperparameters should flow predictably into Auto3DSeg.
* The final generated Auto3DSeg configs should be saved for provenance.
* The pipeline should make it easy to inspect the actual hyperparameters used for a completed run.

If Auto3DSeg requires nested key syntax or template-specific parameter names, that translation should be isolated and tested.

---

# Hyperparameter tuning

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

Important loss-related parameters to support:

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

The exact config syntax can differ, but the design should preserve this level of flexibility.

The pipeline should avoid regenerating identical crops/image stacks repeatedly when only training hyperparameters change.

---

# Inference

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

The output should include:

* per-lesion ROI prediction files
* full-brain prediction file
* machine-readable inference manifest
* failure report for subjects or lesions that could not be processed

The full-brain prediction should preserve the label convention:

```text
0 = background
1 = lesion body
2 = PRL rim
```

---

# Evaluation metrics

For subjects with voxelwise ground truth, the pipeline should compute segmentation metrics.

At minimum:

* lesion Dice
* rim / PRL Dice
* TP, FP, TN, FN for rim voxels
* sensitivity / recall
* specificity
* precision
* negative predictive value
* accuracy
* F1

Optional but useful:

* Hausdorff distance
* 95th percentile Hausdorff distance
* per-lesion PRL detection accuracy
* subject-level PRL count error

Metrics should be saved at multiple levels:

1. per-lesion
2. per-subject
3. per-run aggregate
4. cross-run / grid summary

The implementation should make clear whether a metric is voxelwise, lesionwise, or subjectwise.

---

# Radiomic / geometric feature extraction

The pipeline should include a feature extraction layer for candidate PRL/rim predictions.

Candidate lesions should generally be those with inferred rim voxels near or overlapping the indexed lesion of interest.

Features may include:

* rim voxel count
* rim volume
* lesion volume
* rim convex hull volume
* lesion convex hull volume
* enclosing sphere radius
* PCA-derived shape features
* radial distribution features
* intensity features from phase, FLAIR, T1, or other available sequences
* pyradiomics features, if useful

The design should allow additional features to be added without rewriting the classification pipeline.

Feature extraction should produce a tabular output with one row per subject-lesion-model prediction.

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

# Lesion-level classification layer

The pipeline should include a lesion-level classification layer on top of segmentation.

Purpose:

* The deep learning model may generate candidate rim segmentations.
* A classifier should decide whether each candidate rim should count as a true PRL.

The classification layer should make it convenient to compare algorithms such as:

* logistic regression
* support vector machine / SVM
* random forest
* gradient boosting
* XGBoost or LightGBM if installed
* simple threshold rules
* calibrated classifiers

The design should allow:

* selecting feature columns
* training/test splits
* cross-validation
* class weighting
* imputation
* scaling
* probability outputs
* threshold tuning
* saving fitted models
* applying saved models to new inference datasets

Classifier outputs should include:

```text
subid
sesid
lesion_index
run_id
has_candidate_rim
predicted_prl
prl_probability
classifier_name
classifier_version_or_model_path
feature_set_name
```

The classifier should be usable both from the CLI and from notebooks.

---

# Final classifier-filtered PRL segmentations

After classification, the pipeline should save final full-brain PRL outputs.

Input:

* per-lesion inference predictions
* classifier output table
* subject-level image metadata
* lesion bounding boxes
* selected list of accepted lesion indices

Output:

```text
full-brain classifier-filtered PRL segmentation
```

The final segmentation should include only classifier-approved PRL predictions.

Open design choice:

* Either preserve full segmentation labels:

```text
0 = background
1 = lesion body
2 = PRL rim
```

* Or save a rim-only PRL mask:

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

# Data management and provenance

Every major derived artifact should be traceable to its inputs and parameters.

The pipeline should save metadata for:

* source data root
* subject/session file used
* subject category
* selected rater suffix
* preprocessing parameters
* image stack
* split/fold assignment
* model algorithm
* Auto3DSeg config
* training hyperparameters
* run ID
* inference model/run used
* classifier model and feature set used

Avoid relying on folder names alone for provenance.

Use machine-readable manifests where possible.

---

# Suggested pipeline stages

The implementation does not need to use these exact names, but it should support equivalent stages.

## 1. Stage source data

Input:

```text
ORIG_DATAROOT
subject-sessions.csv
manual PRL label resources
```

Output:

```text
working data root
staging manifest
missing file reports
```

Responsibilities:

* copy or symlink MRI files
* copy or symlink LST-AI outputs
* record missing files
* preserve subject/session identity

## 2. Prepare lesion masks

Input:

```text
space-flair_seg-lst.nii.gz
```

Output:

```text
lstai_lesion_index.nii.gz
lesion metadata table
```

Responsibilities:

* connected-component labeling
* lesion indexing
* lesion count reporting

## 3. Prepare crops

Input:

```text
indexed lesion mask
MRI images
manual labels if available
preprocessing config
```

Output:

```text
per-lesion image crops
per-lesion label crops
bounding box files or manifest
```

Responsibilities:

* bounding box generation
* padding
* boundary handling
* image/label crop generation

## 4. Build dataset manifest

Input:

```text
subject metadata
lesion metadata
label availability
selected rater suffixes
split config
```

Output:

```text
dataset manifest
fold assignment manifest
training datalist
```

Responsibilities:

* PRL vs non-PRL labeling
* fold/test split assignment
* paths to images and labels

## 5. Train segmentation model

Input:

```text
training datalist
Auto3DSeg config
training hyperparameters
```

Output:

```text
trained model folders
saved Auto3DSeg configs
training logs
run metadata
```

## 6. Run segmentation inference

Input:

```text
trained model run
subject data
preprocessing config
```

Output:

```text
per-lesion ROI predictions
full-brain prediction
inference manifest
```

## 7. Evaluate segmentation

Input:

```text
predictions
ground truth labels
```

Output:

```text
per-lesion metrics
per-subject metrics
per-run metrics
```

## 8. Extract features

Input:

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

## 9. Train/apply classifier

Input:

```text
feature table
labels if available
classifier config
```

Output:

```text
fitted classifier
classification report
classified lesion table
```

## 10. Save final PRL outputs

Input:

```text
classified lesion table
per-lesion predictions
subject image metadata
```

Output:

```text
classifier-filtered full-brain PRL segmentation
final subject-level PRL counts
```

---

# CLI and notebook interface

The pipeline should expose a CLI, preferably using Click.

The CLI should be suitable for:

* running full pipeline stages
* launching grid searches
* running inference on one subject or a subject list
* computing metrics
* extracting features
* training/applying classifiers
* saving final outputs

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

The implementation should also be notebook-friendly. Core logic should live in importable Python functions/classes, not only inside CLI functions.

A notebook user should be able to do things like:

```python
dataset = load_dataset("roi_train2")
run = load_run("/path/to/run")
features = extract_features(run, dataset)
model = train_classifier(features)
save_final_outputs(run, model, dataset)
```

---

# Library preferences and design freedom

The implementation may use widely adopted data-science and scientific Python libraries.

Preferred / acceptable libraries:

* `click` for CLI
* `pandas` for tabular manifests and metrics
* `numpy`
* `nibabel`
* `scipy`
* `scikit-image`
* `scikit-learn`
* `MONAI`
* `pyradiomics`, if needed
* `joblib`, `concurrent.futures`, or multiprocessing for parallelism
* `tqdm` for progress bars
* `loguru` or standard `logging`
* `attrs`, `dataclasses`, or `pydantic` for configuration objects
* `pyyaml` or `omegaconf` for configs

Suggested guidance on config/modeling libraries:

* `dataclasses`: good default, minimal dependency, simple.
* `attrs`: good for lightweight immutable configs and validation.
* `pydantic`: good if strong validation, serialization, and user-facing config schemas are important.
* `omegaconf` / Hydra: useful for complex ML experiment configs, but may add complexity.
* Plain dictionaries: acceptable internally, but avoid letting unvalidated dicts spread through the whole pipeline.

The implementer may choose, but should justify the choice and keep the config layer consistent.

---

# Design considerations

## Reproducibility

Every run should be reproducible from saved configs and manifests.

The pipeline should save:

* config files
* generated datalists
* run metadata
* model hyperparameters
* feature-set definitions
* classifier model metadata

## Idempotency

Pipeline stages should be safe to rerun.

Each stage should have clear behavior for:

* skip existing outputs
* overwrite existing outputs
* regenerate missing outputs only

## Scalability

The data root may live on network storage. Avoid unnecessary repeated filesystem scans or excessive file existence checks over large directory trees.

Preprocessing and inference should support parallel execution over subjects or lesions.

## Extensibility

The design should allow future additions:

* new MRI sequences
* new Auto3DSeg algorithms
* new loss functions
* new classifiers
* new radiomic features
* new output conventions
* new subject categories

## Separation of concerns

Avoid putting all logic in a single script.

Suggested separation:

```text
data staging
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

## Human review

The pipeline should preserve artifacts useful for manual inspection:

* cropped ROIs
* per-lesion predictions
* full-brain predictions
* classifier probabilities
* rejected candidate PRLs
* subject-level summaries

Optional future feature: generate ITK-SNAP workspace files or equivalent visualization helpers.

---

# Important open design questions

The implementer should make explicit choices about these:

1. Should staged data be copied, symlinked, or hard-linked?
2. Should all subjects live under one working data root, or should training/inference/count-only subjects be separated?
3. Should crops be stored inside each subject folder, or in a separate derived-data cache?
4. How should different crop/image-stack parameter combinations be named?
5. How should multiple manual label suffixes per subject be represented?
6. How should classifier-filtered outputs encode lesion body vs rim?
7. Should final outputs be saved in subject folders, run folders, or both?
8. What is the stable unique ID for a lesion: `(subid, sesid, lesion_index)` or something else?
9. How should failed subjects or failed lesions be recorded?
10. How should grid-search results be summarized across runs?

---

# Expected deliverables

A good first implementation should provide:

1. A documented config format.
2. A CLI with clear pipeline stages.
3. Importable Python APIs for notebook use.
4. Dataset and run manifests.
5. LST-AI reuse / missing-file reporting.
6. Lesion indexing.
7. ROI cropping with tunable padding.
8. Image-stack creation with tunable sequences.
9. MONAI Auto3DSeg SegResNet training.
10. Grid-search support.
11. Inference on fresh subjects.
12. Segmentation metrics for labeled cases.
13. Feature extraction for predicted rims.
14. Classifier training/application.
15. Final classifier-filtered full-brain PRL outputs.
16. Basic logging and error reports.

---

# Suggested prompt to give Codex / ChatGPT

```text
I want you to design and implement a fresh PRL segmentation and classification pipeline from this specification. Do not assume any existing implementation. Start by proposing an architecture, module layout, config schema, data manifests, and CLI commands. Then implement the pipeline incrementally.

Prioritize:
1. clear data provenance
2. reproducible configs/manifests
3. modular design
4. notebook-friendly APIs
5. CLI usability
6. extensibility to Auto3DSeg models and classifier algorithms

Before writing code, identify open design decisions and propose defaults.
```
