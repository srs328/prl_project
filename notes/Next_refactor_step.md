# New Refactoring Plan 

## Refactoring into MONAI's paradigm

Okay lets try something new. I want to conform my prl segmentation pipeline to MONAI's bundles workflow. 

- Bundle tutorials located at: `/home/srs-9/monai/tutorials/bundle`
    - [Online documentation ](https://monai.readthedocs.io/en/1.0.0/bundle.html)
- Also [here](model-zoo/models/segmentation_template) they show a template for segmentation

One reason for this is so my prl segmentation can integrate with the [MonaiLabel](https://github.com/project-monai/monailabel) tool so I could open an image in 3DSlicer or something and automatically infer a new subject and make edits.cd 

The algo templates (like the segresnet one) that I use are in the bundle worlflow. So i'm envisioning creating a new bundle called "prlsegresnet" which will copy everything from `/home/srs-9/Projects/prl_project/training/algorithm_templates/segresnet` with the addition of "scripts/preprocessing.py" and an appropriate "configs/preprocessing.yaml". Let's do all this without touching anything in src; this bundle should stand alone. We'll have to copy and port a lot of the logic in src as described below.

### Major changes to data handling

The preprocessing module should use MONAI core's Dataset interface (/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/data). We can also use their transforms interface (/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/transforms) for the ROI cropping.

The bundle should receive paths to full size images and labels (so one per subject). Then preprocessing will do the ROI creatinon based on configs loaded by MONAI's ConfigParser. Assigning subjects to folds and train/test will have to be done carefully to see if all the ROI's of a particular subject are part of the same fold/split while still distributing PRL as evenly as possible among folds and testing. 

### Answers to Questions

>cropping
- I think MONAI does have deterministic cropping: monai.transforms.Crop and monai.transforms.SpatialCrop

>Should bounding boxes still be pre-computed with FSL, or computed in-pipeline with pure Python (replacing fslstats)?
- If there were a clean way compute bounding boxes in pipeline with pure Python (using any appropriate library to streamline ease of use e.g: MONAI, nibabel, nipype interfaces like nipype.interfaces.fsl) that would be nice, otherwise subprocess calls to fsl in pipeline are fine too

>Your note explicitly says fold assignment should be per-subject (all lesions from one subject in the same fold) — that's a change from current per-case assignment
- Yes that would be a change from the current method. It's a change I want to explore but is not necessary to commit to yet; other aspects of the pipeline can still be built
- [ ] Try creating a standalone function I can play around with first: for testing, we can create dummy data by defining N subjects, randomly giving them a PRL number (e.g. from 1 - 8, strong right skew e.g. median=2 IQR=\[1,2\]) and a lesion count (my real subjects have percentile,value: 0,3; 0.1,7, 0.25,20; 0.5,27; 0.75,52; 0.9,85; 1,106), then see how well we can evenly distribute PRL's

>PRL label construction (lesion + ring → labels 1/2) — transform or offline step?
- Let's keep the original lesion segmentation offline for now but have a spot where I could integrate it in. Right now I run LST-AI to get the binary lesion masks. Soon I will be producing the pmaps from LST-AI and custom thresholding the probability.
- For rims, label=2 should still be extracted from the prl segmentations and overlaid on the LST-AI lesion segmentation

>How much of the 26K-line Auto3DSeg segmenter.py to keep vs. replacing with MONAI's standard SupervisedTrainer?
- Let's keep segmenter.py exactly as is right now
- First step is to see if we can build prlsegresnet around segresnet without changing segresnet internals

>Inference needs a "fan-in" (reassemble per-lesion predictions back to full brain) for MonaiLabel
- I already have the logic for "fan-in" in `/home/srs-9/Projects/prl_project/src/scripts/inference.py`
- The question is: keep segresnet infer script as is and create my own "prlsegresnet/scripts/postprocess.py" which handles the uncropping, or create a custom "scripts/infer.py" which handles both 

#### My own questions

- My preprocessing produces intermediate files like the lesion index folders; cropped inference labels which must be "fanned-in". It would be good for testing to keep these, so check how this can be handled with MONAI nativel

#### Next step

- Look into all those questions
- Double check the monai.transforms.Crop and monai.transforms.SpatialCrop: if it turns out these are deterministic like I thought, it means you missed it, in which case lets move forward in steps just like we did now where you compile a list of questions and concerns, then I look into each and answer. And try to present your questions and concerns in batches like you just did as much as possible instead of pausing thinking to prompt me

### Planning next step

#### Data Handling

The steps above have been completed according to the plan at `/home/srs-9/.claude/plans/lovely-dreaming-valley.md`. Now I want to take careful steps towards making the prlsegresnet bundle runnable end to end (e.g. with AutoRunner).

Accomplishing this will probably require modification to the original segresnet scripts, but I want to make these modifications as minimal as necessary for this to work. So lets go step by step and see how to do this.

One issue is data loading. If I want to feed full subject images and segmentations, we'll need to write custom handling to produce ROI's within existing infrastructure. `preprocessing.py` has some building blocks for this, but we need to put them together.

Lets say we have a datalist with {"testing": [...], "training": [...]} with paths to the full sized images and labels and no fold assignments. We could load the data similarly to what I tried here:

```python
from core.dataset import Dataset as MyDataset
from monai.data import DatasetFunc, ImageDataset

my_dataset = MyDataset("roi_train2")

img_list = []
seg_list = []
for subid in my_dataset.subjects:
    subject = my_dataset.subject(subid)
    img_path = subject.dir / "flair.phase.nii.gz"
    seg_path = subject.dir / "prl_seg_def_prob.nii.gz"
    if img_path.exists():
        img_list.append(img_path)
        seg_list.append(seg_path)

img_dataset = ImageDataset(
    image_files=img_list,
    seg_files=seg_list,
    image_only=False
)
```



---

## Refactoring analysis code in src/scripts and 

My performance metric code is pretty unwieldy and not easily extensible. Right now it is split over scripts.compute_performance_metrics.py, scripts.analyze_mlflow_runs.py, and scripts.compile_run_metrics.py. 

### The mess of older analysis modules  

compute_performance_metrics and analyze_mlflow_runs are older files. I created them with Haiku before I had a good idea of the shape of this project. I don't use their cli interface because it's too much to try to remember how to pass all the parameters. But compute_performance_metrics::{compute_casewise_stats(), compute_derived_metrics(), get_confusion_matrix()} and analyze_mlflow_runs::{analyze_unified_mlruns, aggregate_metrics} are still used by compile_run_metrics.py. compute_performance_metrics::analyze_dataset is only ever called by experiment.Experiment.evaluate(), and I don't like Experiment.evaluate in the first place. 

### Improvements are in compile_run_metrics()

I've been working on compile_run_metrics.py recently, and it's a step in the direction I want. I reimplemented the compute_performance_metrics::analyze_dataset() logic into compile_run_metrics::performance_metrics(). compile_run_metrics.py is much more extensible in the way that compile_all_metrics() can take any appropriate analysis function as an argument (e,g, mlflow_metrics() and performance_metrics()). I also spent a lot of time making sure it'd easily handle both ExperimentGrid and Experiment. 

#### The annoying part of compile_run_metrics()

However: the compile_run_metrics::load_or_cache_run() function is big issue in the way of readability, usability, and extensibility. Originally it was defined to cache the results of scripts.analyze_mlflow_runs.analyze_unified_mlruns() since it was a heavy computation. But then when I implemented compile_run_metrics::performance_metrics(), I put the compute_casewise_stats function call in there too since that is also a heavy process. So effectively, load_or_cache compiles the data structures that are necessary for these other functions, and also caches them. I feel like those three tasks should be separate.

Say I want to define a new metric column like "fp_per_100_cases" or what have you, and say that I'd need to define a whole new function to compute it: there should be a way to compose that with the output of other functions.

### How to do smart pipeline design?

I would like to unify all of this into a better interface. Try to consider the best principles and practices of pipeline design. And don't feel beholden to the scripts/ folder. I feel like there could be a better home for analysis interfaces.

#### Image analysis interface 

The second analysis interface I'm envisioning is lesion_diagnostics.py. `src/scripts/lesion_diagnostics.py` is in a rough state, but `notebooks/lesion_diagnostics.ipynb` has the working logic that I want for an  image analysis interface. Maybe worth considering a new subfolder under src/ where stuff like this can go. LIke one module for run performance metrics, an other for image and label analysis.

## Revisiting the core modules

I think it would make sense to revisit core because that could be a prerequisite to improving usability of analysis interfaces. I added a lot of small helper functions inside the Dataset class and Experiment class, but I think that's a sign of some design flaw. There must be a way to design Dataset and Experiment so that there are fewer top level entry objects I need to think about specifying.

#### Strenghts of Dataset and Experiment

I like how I can get access to the entire datalist of subjects with:

```python
dataset_name = "roi_train2"
dataset = Dataset(dataset_name)
```

Also, an entire experiment's location can be specified just by having a Dataset object and an experiment id (i.e. roi_train2/stage/run or for single runs, roi_train2/run will work properly as wel)

#### Weaknesses

Dataset does not actually have fully realized paths. If I'm trying to analyze image data, I often have to define an Experiment object just to resolve paths easily. I almost feel like there should be a baseclass like DatasetTemplate while Dataset can function almost like a pandas DataFrame to make it easy to look up subject and lesion_index combinations. It would be cool if DataSet also had functions to load data (like niftis) for me. And maybe Dataset should handle preprocessing configs and Experiment should handle train time configs; but that's a tough choice because expand_xy and expand_z are hyperparameters too. There must be a good compromise,

### Other Notes

- Don't worry too much about cli() right now. That's a different mess for another day.
- I created a new branch for this so feel free to restructure as aggressively as you see fit
- The goal is to simplify, to reduce the number top level parameters I have to worry about passing or defining, and to be extensible so I can add new data (like if I decided on a new radiomic feature to add to the image analysis interface)
- I do most of my work in notebooks rather than the terminal, so I'm looking for modules I can import and easily use like building blocks