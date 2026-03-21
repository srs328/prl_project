# Refactor Plan

## Notes

I'll summarize the current structure then talk about why it's confusing and how maybe it could be refactored with OOP patterns.

### Scripts and preprocessing

preprocessing/ has three scripts that depend on the label_config.json

1. create_rois.py
2. create_datalist.py
3. prepare_training_data.py

Moreover, these three are necessary to run everytime a change is made to the bounding box parameters (e.g. expand_xy, expand_z) during a hyperparameter optimization

scripts/ has generate_experiments.py and launch_experiments.py which are relevant for hyperparameter optimizaiton and setting key monai_config.json parameters

### The training folders

In ../training there are subfolders that correspond to datasets. Meaning roi_train1 was when I had fewer labels, roi_train2 is my current working dataset which is improved. Then the various test_trains was when I was trying to train on full brains. 

These folders contain the datalists, config default templates (monai_config.json and label_config.json), as well as train.py. my_autorunner.py is just a copy I made of train.py as not to get confused by Auto3Dseg's `segresnet_0/scripts/train.py`. The datalist_template.json once created should never be changed because it allows every experiment to train on the same. 

It ended up being confusing having both `${PRL_TRAIN_ROOT}` and ../training, but these both do different things, so I should think of another way or place to store config templates and datalist templates for datasets.

### Why I don't like this

It feels messy. So currently, I have to decide on my training name (right now its roi_train2). Then i have to make sure ../training/roi_train2 has a label_config.jsonc containing `"train_home": "${PROJECT_ROOT}/training/roi_train2"`. Then if I need to change the xy and z expansions, I have to edit the ../training/roi_train2/label_config.jsonc with the new parameters, then go into preprocessing/ to call create_rois.py with the path to ../training/roi_train2/label_config.jsonc as an argument. 

If I want to generate experiments, I have create experiment.config in ../training/roi_train2, then go into scripts/ to call generate_experiments.py with the path to ../training/roi_train2/experiment_config.json as an argument. The experiment_config.json also has to define `"train_home": "${PROJECT_ROOT}/training/roi_train2"` that is consistent with whats in label_config.json. And then monai_config.json has to make sure it has roi_trian2 as part of `training_work_home` as well. Then train.py within ../training/roi_train2 has to collect all these things carefully.

There must be a way to transition away from this preprocessing/ scripts/ and ../training/ structure into a more unified or organized approach. And one that perhaps uses classes and less configs.

In restructuring, I wouldn't mind using new libraries that could make the task easier. For example, if Mako or Jinja2 would make it cleaner to generate experiment specific configs or even train.py scripts, we should consider it. And in general, if you think any task could be improved with some library I don't have installed, feel free to suggest it. I also like loguru for logging and debugging more complicated pipelines, and I do prefer click to argparse, so if the refactor would require enough upheaval to change the cli handling, I'd prefer to switch. 

Once the current paradigm is lean and clean and extensible with all it's current capability, only then will I want to think about how to incorporate the SSL. The SSL should be an additional module that can be optionally added to an experiment, but the current paradigm should run regardless.

## After Refactor

What changed
New files:

src/core/configs.py — PreprocessingConfig (frozen/hashable) and TrainingConfig attrs classes with validators, evolve() support, and serialization methods
src/core/dataset.py — Dataset class that loads from dataset.yaml, owns subjects/folds, and exposes create_rois(), create_datalist(), prepare_data(), preprocess()
src/core/experiment.py — Experiment class with setup(), train(), predict(), evaluate(), and from_run_dir() for reconstructing from existing runs
src/core/grid.py — ExperimentGrid with generate() (Cartesian product + auto ROI creation) and launch() (local/HPC)
src/cli.py — Click CLI with preprocess, train, grid, predict, metrics subcommands
training/roi_train2/dataset.yaml — Single source of truth replacing label_config + monai_config + experiment_config identity fields
pyproject.toml — Package config with prl console_scripts entry point
Refactored files:

src/preprocessing/create_rois.py — Extracted create_rois_for_subjects() and prepare_rois(data_root=...), eliminated global dataroot
src/preprocessing/create_datalist.py — Extracted create_datalist_template() with all explicit params, eliminated global dataroot
src/preprocessing/prepare_training_data.py — Extracted prepare_training_data() with explicit params, eliminated global dataroot
src/helpers/paths.py — Added YAML support to load_config(), added load_dataset_config()
training/roi_train2/train.py — Thin wrapper delegating to Experiment.train()
Key improvements:

No more train_home duplication across 3 config files — derived from dataset name
No more global dataroot — data_root passed explicitly everywhere
ExperimentGrid.generate() auto-calls create_rois() for new expansion params (the missing HPO logic)
prl CLI installed as console_scripts entry point via pip install -e .