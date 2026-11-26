# Notes

3-9-2026

- Need to separate each lst-ai segmentation into a lesion index
  - `c3d space-flair_seg-lst.nii.gz -comp -o lstai_lesion_index.nii.gz`
- Then I need to update all the PRL indices with the new lst-ai index


## Specs

Goal: train a deep learning model to segment PRL rims from lesion cores using flair and phase (possibly T1 too) images. 

First, T2 lesions were segmented with LST-AI. The lesion masks were reindexed so that each lesion cluster would have its own unique index (using `c3d lesion_map.nii.gz -comp -o lesion_index.nii.gz). 

PRL's have already been identified on a couple hundred participants. Their locations are indexed in a spreadsheet based on the lesion_index.nii.gz file. The subject wide spreadsheet has 20 columns for PRLs (e.g. PRL1_label, PRL2_label,... PRLN_label). If a subject only has 1 PRL, then the columns for 2-N are just empty. These columns have the lesion_index so they can be referenced to their corresponding T2 lesion. The lesion_index for some PRL's might not be accurate, so error checking will be important later on.

Each PRL also has a "confidence" assigned: definite, probably, or possible. For now I will only focus on the definite and probable PRL's. The confidence assignments are in columns called "confidence.{N-1}" (e.g. confidence.1 for PRL2_label). The confidence for PRL1_label is just "confidence". The column "Total PRL" refers just to the total number of definite and probable PRL's. 

The manual PRL segmentations have the lesion (core) labeled with 1 and the rim labeled with 2. Basically, we start with the LST-AI segmentation mask (binary) and then paint over the rims with label 2. Right now the manual segmentations have file names like prl_mask_def_prob_{initials}.nii.gz. Initials can be CH, SRS, or LR. CH is the most experienced rater and LR is the least, so if a label exists for multiple people, the precedence is CH > SRS > LR. Most of the existing segmentations are LR currently. Some of these segmentations have additional labels for central veins (3) and extraneous iron content (4). It'll be best to just turn 3 and 4 into 1 so that there are only two classes. 

I think the easiest way to train the model will be to make an ROI bounding each lesion and just crop that out. That will also serve to increase the training set since there will be multiple ROI's per subejct. 




```
#Loop through each lesion in $lesionmap first

#extract bounding box for this PRL label from $lesionmap file and write to file
temp_BBox=$(fslstats "$lesionmap" \
-l $(bc <<< "${lesion_label} - 0.5") \
-u $(bc <<< "${lesion_label} + 0.5") \
-w | sed 's/ 1 /\n/g' | awk '/./')

#temp_bbox means temporary bounding box, which here I write to a text file to pull later, but you could just use the coordinates after expansion (below) to create the ROI.

echo "$lesion_label $temp_BBox" >> $SUBJECTDIR/prlmontage/PRL_bounding_boxes.txt

#Expansion of bounding box parameters: "xy" will expand by the specified number of voxels on each side in the axial plane
expand_xy=20 #determines how much to expand in axial plane
expand_z=2 #if you want to expand above or below the PRL, can increase this
echo "will expand xy dimension of PRL by $expand_xy"
roi_boundaries=$(echo "$temp_BBox" | awk -v expand_xy="$expand_xy" -v expand_z="$expand_z" '{
printf "%d %d %d %d %d %d\n", $1-expand_xy, $2+2*expand_xy, $3-expand_xy, $4+2*expand_xy, $5-expand_z, $6+2*expand_z
}')

#Then run this command to extract the actual ROI, here I name it phase.box because it was extracted from the phase image initially
fslroi [INPUT_IMAGE] $SUBJECTDIR/prlmontage/$sessionDate/prl${lesion_label}.phase.box.nii.gz ${roi_boundaries}"
```
