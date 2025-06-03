#!/bin/bash

root="/media/smbshare" # edit this to be your smbshare root (whatever the parent of 3Tpioneer_bids is)
# root="/mnt/h"
train_folder=test_train0_segresnet
open_inf=0
training_work_home="/home/srs-9/Projects/prl_project/training_work_dirs"

# -------------------------------------

usage() {
    echo "Usage: [-a] [-h] <subid>"
    echo "  -h          Display this help message."
    echo "  <subid>     Subject ID to open (just four digits)"
    exit 0
}

# parse arguments
script_args=()
while [ $OPTIND -le "$#" ]
do
    if getopts :sh opt
    then
        case $opt in
            s)
                open_inf=1
                ;;
            h)
                usage ;;
        esac
    else
        script_args+=("${!OPTIND}")
        ((OPTIND++))
    fi
done

dataroot="$root/3Tpioneer_bids"
if [ ! -d $dataroot ]; then
    printf "Can't find %s \nedit 'root' at top of script\n" $dataroot
    exit 1
fi

subid=${script_args[0]}

if [ -z "$subid" ]; then
    echo "Error: no subject id"
    exit 1
fi


# get the subject+session root (scan_path)
subj_dir="sub-ms${subid}"

# find the first session
ses_dirs=$(ls "$dataroot/$subj_dir")

ses=()
for ses_dir in "${ses_dirs[@]}"
do
    if [[ "${ses_dir}" =~ ^ses-([0-9]{8}) ]]; then 
        sessions+=("${BASH_REMATCH[1]}")
    fi
done

first_ses=${sessions[0]}

for ses in "${sessions[@]:1}"
do
    diff=$(( $(date -d "$ses" +%s) - $(date -d "$first_ses" +%s) ))
    if [[ $diff -lt 0 ]]; then
        first_ses=ses
    fi
done

ses_dir="ses-$first_ses"
scan_path="$dataroot/$subj_dir/$ses_dir"
lesion_dir="$scan_path/lesion.t3m20"
# ensemble_dir="$root/srs-9/prl_project/training_work_dirs/$train_folder/ensemble_output/$subj_dir/$ses_dir"
ensemble_dir="$training_work_home/$train_folder/ensemble_output/$subj_dir/$ses_dir"
echo $ensemble_dir

segmentations=()
while IFS= read -r line || [[ -n "$line" ]]; do
  seg_path="$lesion_dir/$line"
  segmentations+=("$seg_path")
done < "check_prl_cfg.txt"

if [ "$open_inf" -ne 0 ]; then
  if [ -d $ensemble_dir ]; then
    inf=("$ensemble_dir/*.nii.gz")
    segmentations+=($inf)
  else
    echo "sub-ms$subid was not inferred"
  fi
fi

labels_to_show=("$(basename ${segmentations[0]})")
for seg in "${segmentations[@]:1}"
do
    labels_to_show+=(", $(basename "$seg")")
done
echo "Opening ${labels_to_show[*]}"

itksnap -g "$scan_path/flair.nii.gz" -o "$scan_path/phase.nii.gz" "$scan_path/t1.nii.gz" \
  -s "${segmentations[@]}"
