#/bin/bash

work_dir=$1
cd $work_dir || exit

if [[ ( ! -f lst-ai/space-flair_seg-lst.nii.gz ) && ( ! -f lst-ai/annotated_lesion_stats.csv ) ]]; then
    lst --t1 t1.nii.gz --flair flair.nii.gz --output lst-ai --temp lst-ai/processing  
fi