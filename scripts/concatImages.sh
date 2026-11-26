#!/bin/bash


work_dir=$1
shift 1
out=$1
shift 1

while [[ $# -gt 0 ]]; do
    in+=("$1")
    shift 1
done
echo $work_dir
echo $out
echo "${in[@]}"

cd $work_dir || exit

fslmerge -a "$out" "${in[@]}"

# if [[ ! -f flair.phase.t1.nii.gz ]]; then
#     fslmerge -a flair.phase.t1.nii.gz flair.nii.gz phase.nii.gz t1.nii.gz
# fi

# if [[ ! -f flair.phase.nii.gz ]]; then
#     fslmerge -a flair.phase.nii.gz flair.nii.gz phase.nii.gz
# fi