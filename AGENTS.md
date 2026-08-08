# AGENTS.md

This repository contains a PRL segmentation/classification pipeline for detecting perilesional rim lesions in MS MRI. The code is research-oriented and somewhat exploratory, but the main pipeline lives under `src/`.

## Primary entry point

Use the `prl` CLI defined in `src/cli.py`.

Common commands:

```bash
prl preprocess roi_train2
prl train roi_train2
prl grid path/to/experiment.yaml --launch
prl predict /path/to/run_dir
prl metrics /path/to/run_dir
prl infer /path/to/run_dir SUBJECT --data-root /path/to/data
prl compile roi_train2
```

The CLI uses Click. Avoid adding standalone scripts when the behavior belongs in the main pipeline; prefer adding or updating a `prl` subcommand.

## Project roots and config loading

Path roots are centralized in `src/helpers/paths.py`.

Environment variables:

* `PRL_PROJECT_ROOT`: repository/project root
* `PRL_DATA_ROOT`: root directory containing subject imaging data
* `PRL_TRAIN_ROOT`: root directory for training outputs

Use `helpers.paths.load_config()` instead of raw `json.load()` or `yaml.safe_load()` when loading project configs. It supports JSON, JSONC, YAML, and token expansion for `${PROJECT_ROOT}`, `${DATA_ROOT}`, and `${TRAIN_ROOT}`.

## Core architecture

The current pipeline is organized around these classes:

* `core.dataset.Dataset`

  * Represents a named dataset such as `roi_train2`.
  * Loads `training/{dataset_name}/dataset.yaml`.
  * Owns dataset-level information: subject list, PRL label dataframe, rater suffix mapping, fold count, test split, and `datalist_template.json`.
  * The only pipeline stage it owns is `create_datalist()`, because fold assignments belong to the dataset.

* `core.experiment.Experiment`

  * Represents one training run with fixed preprocessing and training parameters.
  * Owns per-run ROI creation, data preparation, setup, training, prediction, and evaluation.
  * Writes run-local configs and copies the prepared datalist into the run directory.
  * Can reconstruct an existing run with `Experiment.from_run_dir()`.

* `core.grid.ExperimentGrid`

  * Handles hyperparameter/grid-search experiments.
  * Expands Cartesian products of preprocessing and training parameters.
  * Creates run directories, writes manifests/configs, and can launch locally or via HPC/LSF.
  * When preprocessing parameters vary, it prepares unique ROI/datalist combinations once.

* `core.configs.PreprocessingConfig`

  * Frozen attrs config for preprocessing parameters: `expand_xy`, `expand_z`, `images`, `processes`, `dry_run`.
  * Provides suffix helpers such as `xy20_z2` and `flair.phase_xy20_z2`.

* `core.configs.AlgoConfig` / `SegResNetConfig`

  * Attrs-based training config.
  * Training parameters flow into MONAI Auto3DSeg through `to_input_dict()`.
  * `SegResNetConfig` maps architecture fields such as `blocks_down` and `init_filters` into MONAI nested-key syntax.

## Current training pipeline

The high-level training flow is:

1. `prl preprocess DATASET`

   * Loads `Dataset(DATASET)`.
   * Creates or reuses `datalist_template.json`.
   * Creates lesion-centered cropped ROIs using the configured expansion parameters.
   * Stacks selected image channels.
   * Writes the prepared datalist named like `datalist_flair.phase_xy20_z2.json`.

2. `prl train DATASET`

   * Builds an `Experiment`.
   * Creates a new run directory unless `--run-dir` is provided.
   * Calls `Experiment.setup()`.
   * Trains with MONAI Auto3DSeg / SegResNet.

3. `prl predict RUN_DIR`

   * Reconstructs the experiment from saved run configs.
   * Runs fold validation inference into `fold_predictions/foldN/`.

4. `prl metrics RUN_DIR`

   * Computes per-case and aggregate metrics from available inference outputs.

5. `prl grid EXPERIMENT_CONFIG`

   * Loads an experiment YAML/JSON config.
   * Generates run directories for parameter sweeps.
   * Optionally launches locally or on HPC.

## Fresh-subject inference

Fresh-subject inference lives in `src/scripts/inference.py` and is exposed through `prl infer`.

The inference pipeline:

1. Load the trained run’s preprocessing config from `label_config.json`.
2. Generate lesion bounding boxes from `lstai_lesion_index.nii.gz`.
3. Crop each lesion ROI from the subject’s image channels and lesion mask.
4. Stack the image channels.
5. Create `inference_datalist.json`.
6. Run MONAI ensemble inference using the trained folds.
7. Uncrop ROI predictions back into a full-brain volume named `prl_inference_{run_id}.nii.gz`.

Fresh subjects do not have PRL annotation/rater files, so inference preprocessing should not require `prl_mask_def_prob_*` files or rater suffixes.

## Data assumptions

Subject folders live under `PRL_DATA_ROOT` and are usually named like:

```text
sub{id}-{session}/
```

Expected subject-level files include:

```text
flair.nii.gz
phase.nii.gz
t1.nii.gz              # optional/future
space-flair_seg-lst.nii.gz
lstai_lesion_index.nii.gz
```

Training subjects may also have PRL label files such as:

```text
prl_mask_def_prob_<suffix>.nii.gz
```

In PRL label volumes:

* lesion = label 1
* rim = label 2
* other labels may exist but should generally be ignored unless intentionally adding support for them.

Per-lesion cropped outputs are stored in numeric lesion-index subdirectories under each subject folder.

## Run directory assumptions

Training runs live under:

```text
$PRL_TRAIN_ROOT/{dataset_name}/...
```

A typical run directory contains:

```text
label_config.json
monai_config.json
datalist_*.json
mlruns/
segresnet_0/
segresnet_1/
segresnet_2/
segresnet_3/
segresnet_4/
fold_predictions/
ensemble_output/
performance_metrics.csv
```

Do not assume all of these exist before training/prediction/metrics have been run.

## Development guidelines for agents

* Prefer editing code under `src/`.
* Prefer the `prl` CLI over adding one-off scripts.
* Keep dataset-level logic in `Dataset`.
* Keep per-run logic in `Experiment`.
* Keep grid/HPO logic in `ExperimentGrid`.
* Keep config schema changes in `core.configs`.
* Use `attrs.evolve()` when deriving modified config objects.
* Be careful with path handling: many paths are on SMB/HPC filesystems and can be slow.
* Avoid repeatedly validating or stat-ing thousands of files unless necessary.
* Do not hard-code local absolute paths unless they are documented defaults.
* Use environment variables and `${PROJECT_ROOT}`, `${DATA_ROOT}`, `${TRAIN_ROOT}` tokens when possible.
* Preserve compatibility with existing run folders that contain saved `label_config.json` and `monai_config.json`.
* Be cautious around MONAI Auto3DSeg internals; training parameters should flow predictably through `AlgoConfig.to_input_dict()`.

## Known messy/outdated areas

* `notes/Pipeline_Notes.md` contains useful historical context, but parts of it describe an older manual-config workflow.
* `CLAUDE.md` appears closer to the current architecture and should be treated as more up to date.
* Training parameter handling is actively being refactored toward attrs-based algorithm configs.
* Inference support exists, but may still contain rough edges and TODOs.
* Some TODO/FIXME comments in `src` are real design notes, not necessarily stale comments.

## Testing / validation

There is no obvious formal test suite yet. For safe changes:

1. Run targeted CLI commands with `--dry-run` where supported.
2. Test config loading with the intended dataset.
3. For preprocessing changes, verify generated paths and datalist entries before launching training.
4. For training changes, initialize a run with `prl train DATASET --init-only` before running full MONAI training.
5. For inference changes, test one subject before using `--all` or `--subjects-file`.

## Important implementation notes

* `PreprocessingConfig.image_prefix` sorts image names before joining them, so generated stacks are named deterministically.
* `Experiment.setup()` creates run-local configs and copies the prepared datalist into the run directory.
* `Experiment.from_run_dir()` reconstructs an experiment from saved run configs.
* `Experiment.predict()` runs fold validation inference, not fresh-subject inference.
* `prl infer` is the fresh-subject inference command.
* `ExperimentGrid.generate()` deduplicates preprocessing configs so repeated image/expansion combinations are not prepared repeatedly.

```

A couple of repo-specific notes I would add: your `cli.py` command list has drifted slightly from `CLAUDE.md`; for example, `grid` now takes only `experiment_config` as a positional argument, not `dataset_name experiment.yaml`, and `infer` plus `compile` are present in the CLI. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}  

Also, `Dataset.load_config()` currently sets unspecified relative `subjects` / `suffix_to_use` paths to `None` in the `else` branch, which may be intentional but looks suspicious because it also catches missing keys and absolute-path edge cases together. Codex should be cautious there. :contentReference[oaicite:5]{index=5}
```
