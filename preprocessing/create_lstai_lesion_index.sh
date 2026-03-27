#!/bin/bash

root="/mnt/h" # edit this to be your smbshare root (whatever the parent of 3Tpioneer_bids is)
# root="/mnt/h"

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

sesid=${script_args[1]}
echo $sesid

session_root="$dataroot/sub-ms${subid}/ses-${sesid}"
lstai_dir="$session_root/lst-ai"
lesion_mask="$lstai_dir/space-flair_seg-lst.nii.gz"
lesion_index="$lstai_dir/lstai_lesion_index.nii.gz"

echo "Running: c3d $lesion_mask -comp -o $lesion_index"
c3d "$lesion_mask" -comp -o "$lesion_index"
