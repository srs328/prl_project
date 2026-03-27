# Specs

## Context

Goal: train a deep learning model to segment PRL rims from lesion cores using flair and phase (possibly T1 too) images. 

First, T2 lesions were segmented with LST-AI. The lesion masks were reindexed so that each lesion cluster would have its own unique index (using `c3d lesion_map.nii.gz -comp -o lesion_index.nii.gz). 

PRL's have already been identified on a couple hundred participants. Their locations are indexed in a spreadsheet based on the lesion_index.nii.gz file. The subject wide spreadsheet has 20 columns for PRLs (e.g. PRL1_label, PRL2_label,... PRLN_label). If a subject only has 1 PRL, then the columns for 2-N are just empty. These columns have the lesion_index so they can be referenced to their corresponding T2 lesion. The lesion_index for some PRL's might not be accurate, so error checking will be important later on.

Each PRL also has a "confidence" assigned: definite, probably, or possible. For now I will only focus on the definite and probable PRL's. The confidence assignments are in columns called "confidence.{N-1}" (e.g. confidence.1 for PRL2_label). The confidence for PRL1_label is just "confidence". The column "Total PRL" refers just to the total number of definite and probable PRL's. 

The manual PRL segmentations have the lesion (core) labeled with 1 and the rim labeled with 2. Basically, we start with the LST-AI segmentation mask (binary) and then paint over the rims with label 2. Right now the manual segmentations have file names like prl_mask_def_prob_{initials}.nii.gz. Initials can be CH, SRS, or LR. CH is the most experienced rater and LR is the least, so if a label exists for multiple people, the precedence is CH > SRS > LR. Most of the existing segmentations are LR currently. Some of these segmentations have additional labels for central veins (3) and extraneous iron content (4). It'll be best to just turn 3 and 4 into 1 so that there are only two classes. 

I think the easiest way to train the model will be to make an ROI bounding each lesion and just crop that out. That will also serve to increase the training set since there will be multiple ROI's per subejct. 

## Workflow

Right now each subject's folder is called `sub{subid}-{date_mri}` where subid is the 4 digit number from subid column and date_mri is the 8 digit number from the date_mri column of the PRL spreadsheet. There's also an `lstai_lesion_index.nii.gz` and the binary lesion mask which is called `space-flair_seg-lst.nii.gz`. The segmentations will be there too as well as flair.nii.gz, phase.nii.gz, and t1.nii.gz files. 