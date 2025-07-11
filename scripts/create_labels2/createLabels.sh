#!/bin/bash

work_dir=$1

cd $work_dir || exit

if [[ ! -f lesion.t3m20/prl_mask_def_prob.nii.gz ]]; then
    fslmaths lesion.t3m20/prl_labels_definite.nii.gz -add lesion.t3m20/prl_labels_probable.nii.gz -bin lesion.t3m20/prl_mask_def_prob.nii.gz
fi

if [[ ! -f lesion.t3m20/lstai_and_prl_def_prob_tmp.nii.gz ]]; then
    fslmaths lst-ai/space-flair_seg-lst.nii.gz -add lesion.t3m20/prl_mask_def_prob.nii.gz lesion.t3m20/lstai_and_prl_def_prob_tmp.nii.gz
fi