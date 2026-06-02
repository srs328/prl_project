# Training Pipeline

## Goals

Add running list of goals here:

### 03/2026

My goal is to refactor or restructure my training pipeline so that it's easier to run, more extensible to hyperparameter optimization and grid searching, and portable. Right now I run it locally but I want to port it to the HPC after it's ready for hyperparameter optimization.  

- [x] Some way of centrally defining the project root and dataroot prefixes (maybe as environment variables or in a python file like an `__init__.py` or something else) so they can be loaded into every script instead of needing to be hardcoded into the config .json files
- [ ] Everything described in [[Training Pipeline#Hyperparameter optimization]]
- [x] Figure out a method for job submission when this is ported to the HPC (using the IBM LSF platform e.g. bsub and job arrays)

### 06/2026

EDIT: I have been running grid searches on the HPC for awhile now and it does seem to work pretty smoothly. But I'm coming back to this after a break, and it has felt a bit tedious to reorient myself to how everything works, so I want to do a detailed audit to determine whether any small or thorough refactoring would be beneficial before diving back into the work

MOST CURRENT GOALS:
Right now I want to do a proper audit and am open to a deep refactor if warranted. I wonder if my code is spread out over too many modules in certain circumstances. 

- [ ] Be more convenient to compare metrics between experimwents, after radiomic classification, and play around in notebooks
- [ ] Port code from `notebooks/save_final_prl_seg.ipynb` for saving final segmentations for visual inspection, but only after I've done any important refactoring that will make my life easier down the line
- [ ] Conceptually, I want to understand why classification based on deep learning segmentation from roi_train2_t1 (ie roi_train2_t1/run1) is performing much worse compared to roi_train2/stage6_sweep_dicece_wts/run1

## Project Structure

I have three sets of subjects. 24 who have segmented PRL rims (call it the training set); 52 who have PRLs identified to lesions by an index (call it the inference set); a few hundred who just have known PRL counts but no reliable index to label lesions as PRL or not (call it the full inference set) (note that while I could have better, more descriptive names for these three data partitions, those three names are sort of baked into the code already, so I'll avoid changing them until later when I feel like refactoring).

### Segmentations and original data organization

The training data `dataroot` is `/media/smbshare/srs-9/prl_project/data`. It contains the 24 segmented subjects' folders. The names are in the format `sub<subid>-<sesid>`, and right now there is only one session per subject, so every subid appears once. I call each of these `subject_root` in most code

```
sub1010-20180208
sub1011-20180911
```

Before any preprocessing has been done, these folders start with the following structure:

```
sub1011-20180911
├── flair.nii.gz
├── lstai_lesion_index.nii.gz
├── phase.nii.gz
├── prl_mask_def_prob_LR.nii.gz
├── prl_mask_def_prob_SRS_CH.nii.gz
├── prl_mask_def_prob_SRS.nii.gz
├── space-flair_seg-lst.nii.gz
└── t1.nii.gz
```

- `space-flair_seg-lst.nii.gz` is the binary lesion mask produced by LST-AI
- `lstai_lesion_index.nii.gz` is the indexed lesion mask produced by `c3d space-flair_seg-lst.nii.gz -comp -o lstai_lesion_index.nii.gz`
- `prl_mask_def_prob_<initials>.nii.gz` contains the rim segmentation. The initials refer to the rater (CH, SRS, LR; SRS_CH means that CH made fixes to SRS). CH is the most senior rater, then SRS, then LR. "def_prob" means that it only includes PRL's that were identified with a confidence of "definite" or "probable" (~~see [Project Structure](Training%20Pipeline#Project%20Structure) for more information~~ lost this link)
    - Rims are labeled 2, lesion is labeled 1. In some cases, there are additional labels (3=central vein, 4=extraneous iron content; 5=extralesional hyperintensity). All but rim (label 2) are ignored. In preprocessing, the rims are extracted from this image and saved into  "prl_rim_def_prob_\<rater\>.nii.gz" and overlayed onto `space-flair_seg-lst.nii.gz` to get the actual training segmentations. Eventually, we may add the central vein sign if rim segmentation is successful.
- There are three imaging modalities. This will be a hyperparameter. For now, just flair and phase are being used; I have done training tests with T1 added to the stack as well

### Code Workspace

This is subject to change because I want it to be more organized. In my projects folder, I have `$HOME/Projects/prl_project`. 

```
put directory structure here when ready
```

The main pipeline code lives under `src`. However there was a time when I had the code at the top level, and so some resource files live in the top level `resources` folder and I haven't migrated all those references to `src/resources`, nor have I copied all the files from the former to the latter.

#### Some notes about legacy top level folders

- `resources`
	- `labels_to_use.csv` contains the suffix (initials of the rater) of the PRL label file that is to be used for training
	- `PRL_spreadsheet-lstai_update_label_reference.csv` maps each PRL to its corresponding index in `lstai_lesion_index.nii.gz`. Each subject gets a row ("subid"). The column "date_mri" contains the sesid (so `f"sub{subid}-{date_mri}"` names the `subject_root` folder)
		- The columns `PRL<i>_label` contain the index for `lstai_lesion_index` 
		- The columns `confidence.<i-1>` label the confidence about whether the identified PRL really is a PRL (these were professionally rated by neurologist CH and his neuroradiologist collegue a few years ago for a different project). The three values are `["definite", "probable", "possible"]`. For now I am only using the definite and probable labels.
		- The column `Total PRL` refers to the total number of PRL with definite or probable ratings
	- `PRL_spreadsheet_label_reference.csv` is an old label reference from outdated lesion indexes. The script `preprocessing/convert_lesion_index.py` is what I used to convert this to `PRL_spreadsheet-lstai_update_label_reference.csv`
	- `PRL_labels.csv` is a master reference with many more subjects and columns of notes. Only the subjects whose PRL's I could reconcile with the old lesion index are included in `PRL_spreadsheet_label_reference.csv`
		- The rest of the CSV's and excels are variations on the above that are either incomplete or unimportant
	- `subject-sessions.csv` is a master list of all subjects and their baseline MRI session (which corresponds to "date_mri" in the PRL label references)
	- `subjects.txt` is a running list of all subjects who have at least one manual label (i.e. `prl_mask_def_prob_<initials>.nii.gz`

- `preprocessing`: right now this contains code that used to process the raw `subject_root` folders to create the cropped images and labels for training and to copy subject data from my original data location to my prl working directories. I don't think anything within this top level preprocessing folder is called from within the pipeline, but I still might depend on some of these scripts if I needed to run a particular copying task on entirely new data (but I haven't run into that situation in awhile so I don't know which of these files I may need to keep for the future, so I will just keep all of them for now) 
- `training`: This folder is still very important. It contains training runs and experiment specific folders with the customized configs and code for running that particular session. ~~I'm trying to think of a way to more seamlessly integrate preprocessing with the stuff here~~ EDIT 06/2026: I think I did improve integration of preprocessing with stuff here a lot. The basic conceit is: a folder like roi_train2 contains a datalist_template for a particular set of subjects (e.g. the 24 training subjects) with already defined train/test split. These assignmentsm are in datalist_template.json. For particular training parameters like crop padding, datalists are created from the template. These derived datalists will have full paths to images and labels while the template will have prefixes. `dataset.yaml` contains defaults for that training paradigm which can be overriden for specific HPO experiments. `roi_train2_t1` is a copy of `roi_train1` for the three image stack (flair, phase, t1). At the time that I made it, it seemed easier to just create a new paradigm folder instead of trying to integrate the 3 image stack via overrides. `inference_dataset` and `full_inference_dataset` were for the other two sets of subjects. These don't have ground truth. I added these in after writing all the core code, and so I ended up needing to go back and modifying the core code to be compatible with datasets missing ground truth, and it possibly got a bit hacky here and there. 

#### Production code in src

`src/cli.py` is the entry point for anything production level. `src/analysis` contains modules for various post deep learning steps like gathering performance metrics or doing radiomic classification. `src/helpers` has many important helper modules. `src/resources` has a few things that aren't referenced much in code but which I still need sometimes. `src/hpc` has scripts needed for running training on the HPC. `src/preprocessing` is for preprocessing labels and images before deep learning. This includes creating image stacks and cropping segmentations

### Training work directories

The home location for MONAI training outputs is `training_work_home` at `/media/smbshare/srs-9/prl_project/training`. 

My pilot session is in `$training_work_home/roi_train1_segresnet`. This session was run on a smaller training set with rougher PRL labels and an older verion of the code. The MONAI outputs are directly under `roi_train1`.

My current working session is in `$training_work_home/roi_train2`. I am intending this to be where all or most of the training experiments I run with the updated training set will live. MONAI outputs will be contained in subfolders here. Right now I've already produced `roi_train2/run1` and `roi_train2/run2` using the most current version of my pipeline. It looks like

```
roi_train2/run2
├── algorithm_templates
├── cache.yaml
├── datalist_xy20_z2.json
├── datastats_by_case.yaml
├── datastats.yaml
├── directory_structure.txt
├── info.txt
├── input.yaml
├── monai_config.json
├── segmentation_config.json
├── segresnet_0
├── segresnet_1
├── segresnet_2
├── segresnet_3
└── segresnet_4
```

## Pipeline

### Pre-preprocessing

The original data lives somewhere else entirely and gets copied to `$dataroot` by `preprocessing/copy_files.py`. Everytime someone creates a new PRL segmentation, I will update `resources/subjects.txt` and then run `copy_files.py`. I feel most comfortable with manual intervention here, but I should maybe incorporate the configs below into this code so I don't have to edit dataroot in multiple places.

After moving on from `roi_train1_segresnet` to `roi_train2`, I renamed the the former's dataroot folder to data0 and created a fresh dataroot for the latter. The old subjects list for roi_train1 is in `training/roi_train1/subjects.txt` so I don't forget.  

### Configs

#### LEGACY DOCUMENTATION

A lot of the description below may no longer be relevant since I updated config handling. I haven't properly documented the new approach yet

`label_config.json` contains parameters necessary to prepare all the segmentations and MRI images before MONAI sees them

- `images`: which images to stack. Eventually I will try adding T1 to the stack
- `subjects`: I manually created the folder `training/roi_train2` and copied `subjects.txt` into it so that I'd preserve the subjects list for this in case the original running list is updated
- `suffix_to_use`: specifies which prl label file to use since there may be multiple. If this isn't specified, I have a default priority list incorporated into the code, but right now I do specify a suffix for every subject
- Expansion paramters: these are to become tunable parameters for a grid search to determine the optimal cropping. Right now my pipeline only supports constants, but I may consider adding a relative expansion (maybe small lesions should get a greater expansion than larger lesions so all the cases have more consistent sizes; or maybe one could argue larger lesions need a larger expansion, I don't know)
	- `expand_xy`: how much to expand the original bbox for each lesion in the xy plane
	- `expand_z`: how mcuh to expand the original bbox for each lesion in the z axis  
- `images`: which images to stack for the model. This will need to be a tunable parameter eventually

```json
{
    "dataroot": "/media/smbshare/srs-9/prl_project/data",
    "train_home": "/home/srs-9/Projects/prl_project/training/roi_train2",
    "prl_df": "/home/srs-9/Projects/prl_project/PRL_spreadsheet-lstai_update_label_reference.csv",
    "subjects": "/home/srs-9/Projects/prl_project/training/roi_train2/subjects.txt",
    "suffix_to_use": "/home/srs-9/Projects/prl_project/training/roi_train2/labels_to_use.csv",
    "expand_xy": 20,
    "expand_z": 2,
    "images": ["flair", "phase"]
}
```

`monai_config.json` contains parameters relevant to the train test splitting and Auto3dSeg

```json
{
    "training_work_home": "/media/smbshare/srs-9/prl_project/training/roi_train2",
    "N_FOLDS": 5,
    "TEST_SPLIT": 0.2,
    "train_param": {
        "algos": ["segresnet"],
        "learning_rate": 0.0002,          
        "num_images_per_batch": 1,       
        "num_epochs": 250,                
        "num_warmup_epochs": 1,          
        "num_epochs_per_validation": 1,
        "roi_size": [44, 44, 8]
    }
}
```
~~
- `training_work_home`: the top level folder that contains a set of experiments. For now I will classify a "set" of experiements as those which contain the same training subjects and labels
- `TEST_SPLIT` is the percent to hold from training so I can do inference and compute Dice
- `train_param` is the dict I pass to `Autorunner.set_training_params`. I want to create the ability to set these in such a way that I could potentially do a gridsearch
	- There may be more MONAI parameters I could customize as I look more deeply into it

It would be nice to find a way that I could use relative paths in these configs and set an environment variable or some `__init__.py` approach to setting the project and dataroots so I could port this all over to the HPC more easily.

===END LEGACY CONFIG DOCUMENTATION===

#### Hyperparameter optimization

EDIT (06/2026): I think I accomplished the goals listed below already

I need to think of a good interface for me to use to vary the parameters in `label_config.json` and `monai_config.json`. At some point maybe I'd need code to automatically generate these two configs and save them into appropriate locations (like subfolders under `training_work_home`). And maybe a master controller config file where I set the paramter space to vary and pass to a script that generates downstream scripts to run everything. ~~At somepoint I'll read [tutorials/auto3dseg/docs/hpo.md at main · Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/docs/hpo.md) to see if it can handle the `monai_config.json` part of it and then get some ideas for how I can handle the `label_config.json` side of it.~~ (determined that the auto3dseg hpo would be more effort than it's worth, possibly overkill)

### Pipeline (ALSO MOSTLY LEGACY but helpful info still)

~~Below I'll describe how I run everything under the current paradigm. Any part of this pipeline is subject to change if there is room for improvement based on what I have in [[Training Pipeline#Goals]]~~

The description below is out of date in terms of how I actually initiate the steps, but it still accurately describes the key concepts. My current approach all starts in `src/cli.py` and is mostly implemented there and in `src/core` modules.

#### Step 0: 

Right now I manually create the two main config jsons inside of `training/roi_train2` with the desired parameters for a run.

#### Step 1: Create cropped ROI's

I run it with

```bash
python preprocessing/create_rois.py training/roi_train2/label_config.json --processes 5
```

Loops through the subjects and calls `prepare_roi(subid, suffix, prl_df, expand_xy, expand_z)` on each. 

1. Calls `preprocessing/define_bounding_boxes.sh` on the `subject_root` folder with expand_xy and expand_z as additinoal arguments. 
	- Looks at `lstai_lesion_index.nii.gz`, creates the smallest ROI to contain each lesion, and saves the bboxes to `lstai_bounding_boxes.txt` 
	- Then applies the expansion parameters to each bbox and saves to `lstai_bounding_boxes_xy${expand_xy}_z${expand_z}.txt`
	- TODO: just realized my setup is inefficient because it produces `lstai_bounding_boxes.txt` every single time I run it with new expansion parameters, even if it already exists
2. Loops through the subject's lesion index. Creates a folder `subject_root/<index>` for each, uses `fslroi` to crop and save the MRI images into the index folders. 
	- Every crop is saved with suffix `_xy{expand_xy}_z{expand_z}.nii.gz`. 
	- Also saves a crop for `space-flair_seg-lst.nii.gz` as `lesion_xy{expand_xy}_z{expand_z}.nii.gz`. 
	- If index corresponds to a PRL, then the PRL rim segmentation will also be cropped:
		- First call `ensure_ring_seg(subject_root, suffix=suffix)` to make sure that `prl_rim_def_prob_{suffix}.nii.gz` exists. If not, create it by extracting the rim label from `prl_mask_def_prob_{suffix}.nii.gz` (`fslmaths {full_seg_path} -thr 2 -uthr 2 {ring_seg_path}"`)
		- Crop and save the ring segmentation into the index's folder as `prl_rim_def_prob_{suffix}_xy{expand_xy}_z{expand_z}.nii.gz`
		- Combines the cropped ring and the lesion by calling `prepare_prl(lesion_path, rim_path, output_path)` where output_path is `prl_label_{suffix}_xy{expand_xy}_z{expand_z}.nii.gz`

This script requires `label_config.json` to get the subjects list, the suffixes\* to use, and the path to `PRL_spreadsheet-lstai_update_label_reference.csv`. 

\*There is the capability of choosing a default suffix based on seniority of the rater, but seniority is not a reliable criteria currently because of how our labeling paradigm changed over the past year; so right now every subject needs a reliable suffix assigned by me. 

#### Step 2: Create the datalist template

```bash
python preprocessing/create_datalist.py training/roi_train2/label_config.json training/roi_train2/monai_config.json
```

This step produces a template datalist that should be reused on all experiments to keep the partitioning of cases to folds and testing consistent. This step also ensures that PRL cases are evenly distributed across folds and in the training and test set. This script needs the `monai_config.json` for the number of folds and the and train/test split. It needs `label_config.json` to get the PRL dataframe and the suffixes for PRL's. The template datalist is saved like this so that all that's needed in later steps is to tack on the expansion parameter. 

```json
{
	"subid": 1076,
	"lesion_index": 16,
	"fold": 0,
	"image": "/media/smbshare/srs-9/prl_project/data/sub1076-20170912/16/flair.phase_",
	"label": "/media/smbshare/srs-9/prl_project/data/sub1076-20170912/16/prl_label_CH_SRS_"
},
```

#### Step 3: Prepare the final datalist for an experiment

```bash
python preprocessing/prepare_training_data.py training/roi_train2/label_config.json
```

Goes through the datalist template, tacks on the expansion parameter suffixes, and ensures that all the files exist. It will create the stacked image from `label_config['images']` if it does not exist for the given expansion parameter.

#### Step 4: Start AutoRunner

```bash
python training/roi_train2/train.py
```


### Notebooks

At the top level in `notebooks` there are many jupyter notebooks. Ignore all except for `notebooks/evaluate_inference2.ipynb` and `notebooks/save_final_prl_seg.ipynb`. These are the only one I have been working with recently and it essentially describes how I've been interacting with my project recently and probably contains insights into how things can improve, what types of hacks I'm using due to inelegencies, etc. The most recent code I added to src is classification code which I prototyped in evaluate_inference2. save_final_prl_seg contains some cells for saving radiomic filtered segmentations which I have yet to port into module level code that can be run from the cli.

## Training Run Examples

in `/media/smbshare/srs-9/prl_project/training/roi_train2`, any subfolder from stage3 till stage8 is probably a good example. Stage6 maybe best since it is stage6/run1 which I am currently using as my best run for subsequent radiomic classification

ALso in `/media/smbshare/srs-9/prl_project` are the folders `full_inference_data` and `inference_data` which contains inference results for the other two subject sets. 

## Scratch 

great it works. I just had to fix a small error, which was to add "run" right after infer: 

cmd = [
            sys.executable,
            str(infer_script),
            "run",
            f"--config_file={config_file}",
            f"--infer#output_path={fold_output_dir}",
            f"--data_list_file_path={temp_datalist}",
        ]

One more thing to make my life easier as I try to inspect the fold labels visually. I already wrote this script to create commands to open itksnap to view images and labels. Im working through a vscode tunnel, so I cant directly open GUI's, so I just have it output a list of all the commands I could want, then I can just copy and paste them. 

This current file just has 