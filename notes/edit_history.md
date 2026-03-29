## New Predictions

The next step: use the trained model to infer on completely new subjects that aren't in the train or test data. What I envision this to entail:

1. First I decide on which run to use to infer
2. A fresh subject will just have a flair.nii.gz and a phase.nii.gz (for now; t1.nii.gz later, so the "images" key from the selected run will be used to determine), and also the lstai_lesion_index.nii.gz and the space-flair_seg-lst.nii (see "notes/Pipeline Notes.md for a refresher if needed).
3. Preprocessing needs to be run on this subject: I think just create_rois.py for this subject with the bounding box parameters pulled from the selected run should do the trick. These subjects will not have any prl_mask_def_prob_ files btw, so "suffixes" referring to the rater's initials are not applicable. Worth double checking that create_rois.py will work properly and not do any unnecessary work for the task. If itd do unnecessary work or fail too badly, could consider a separate function? But itd be nice to have one function thatd work for both purposes (training pipeline and new inference pipeline).
4. A mock datalist (lets call it "inference_datalist.json" which lives inside the subject's folder) will have to be created so that each of this subject's ROI's is seen by the model as a case to segment. So the current "create_datalist.py" will not work, but perhaps a new function can be added to it like "create_inference_datalist.py" for this purpose. It will essentially create one datalist for one subject. The paths should be relative. This whole process should default to the regular dataroot, but be able to get an override passsed. I will definitely be overriding it for this (ill figure out some spot for inference data that can be my inference dataroot)
5. At some point image stacks will have to be created. I wonder if `prepare_training_data.py` should be renamed to `prepare_data.py`, and it should have a new function called `prepare_inference_data()` or something to handle the inference_datalist.
6. Model inference should be run on this datalist with the output of ensemble prediction set to the same subject folder and the outputs having the suffix "infer" postpended. I know MONAI lets you specify these things
  - Found it, here's their code, all the parameters it takes are in there: /home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/apps/auto3dseg/ensemble_builder.py.
  - At the very least we'll need to use data_root_dir, output_dir, and output_postfix. The ensembling that autorunner does right after training saves output with the same file structure as the input labels. That's what I want for this too, and we should verify whether it will default to the same or not. ensemble_builder.py has on line 508 "separate_folder": kwargs.pop("separate_folder", False)", so we should definitely make sure
  - Awhile back I tried creating a library for auto3dseg training for myself (full brain tasks, so very different and less complicated than now). It's located at "/home/srs-9/Projects/ms_mri/src/monai_training". It has its flaws because I wrote it myself instead of with any help from AI agents, but the file inference.py is how I did it in the past. We could probably do it better now, but it could be a reference if needed.
7. By this point, every lesion folder for the subject should have a file like `lesion_{expand_params}_infer.nii.gz` (in addition to the original `lesion_{expand_params}.nii.gz` and the cropped image files). The final step is to uncrop each of the inference segmentation ROIs. Sonnet couldn't find a super straightforward way and suggested the following approach which I confirmed does work (for one ROI)

```
python -c "
import nibabel as nib, numpy as np
orig = nib.load('./flair.nii.gz')
out = np.zeros(orig.shape, dtype=np.uint8)
crop = nib.load('./1/rim.nii.gz').get_fdata().astype(np.uint8)
xmin,xsize,ymin,ysize,zmin,zsize = <pull these from the right bounding box file>
out[xmin:xmin+xsize, ymin:ymin+ysize, zmin:zmin+zsize] = crop
nib.save(nib.Nifti1Image(out, orig.affine, orig.header), './rim.nii.gz')
"
```

But what I ultimately want is for all the inferred ROI labels for the subject to end up on the same file. See what would be the most resource and time efficient way to accomplish something like this. In the end, the final output should be at the top of the subjects folder (where the flair and phase and other images live) called "prl_inference_{some identifier for the run used to make it}"
- Run identifiers is something worth talking about. Right now we identify them based on their subpath inside of the dataset dir (e.g. roi_train2 has runs like run1, run2, but also stage1_crop_lr_sweep/run1 is an identifier). This is fine for now, we could name an inference "prl_inference_stage1_crop_lr_sweep_run1.nii.gz"

## Cleaning up and refactoring the train params workflow (marked as complete)

Right now the way that train params are handled is really messy and hard to follow. Its exacerbated by the fact that segresnet does its own post processing in some cases. Like how the pipeline looks to see a particular dataset's default config (fine for the PreprocessingConfig, bad for TrainingConfig). Then it looks for overrides for a particular train run. Then it has to evolve the TrainingConfig and try to incorporate it with segresnet.

But there's an obvious fix. I just realized: TrainingConfig should just be an attrs representation of the entire hyper_parameters.yaml from the algorithm_templates (I saved this folder in ./training for our reference). The defaults will just be the template defaults. Then it's easy and predictable to evolve them. 

What I'm envisioning is a parameters base class so that I could easily try out swinunetr or dints later. The configs for those two networks is a bit more complicated because they don't have just one .yaml file in `configs/`. I cloned the auto3dseg tutorial from github into /home/srs-9/monai/tutorials/auto3dseg so you can analyze the idiomatic use of auto3dseg models to inform your plan. 

Can you come up with a plan for reworking the handling of training params like this. I'm aiming for predicability and ease of use; no more looking in several locations to follow the flow of parameters. 

Also just for fun, after you've analyzed, tell me if python abstractbaseclasses are useful here or if it's overkill. I dont have a good working understanding of how they work and when you'd want those over a straightforward base class and inheritance, so itd be cool to learn.

