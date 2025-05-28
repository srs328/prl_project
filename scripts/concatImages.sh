#!/bin/bash

work_dir=$1

cd $work_dir || exit

if [[ ! -f flair.phase.t1.nii.gz ]]; then
    fslmerge -a flair.phase.t1.nii.gz flair.nii.gz phase.nii.gz t1.nii.gz
fi