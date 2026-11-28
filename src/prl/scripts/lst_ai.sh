#!/bin/bash

source /home/srs-9/.virtualenvs/lst-ai/bin/activate

work_dir=$1
cd "$work_dir"

log="$work_dir/run_lst_ai.log"

# Open FD 3 for the subject-specific log
exec 3>>"$log"

info() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" ; }

run_if_missing() {
    out="$1"
    shift
    if [[ -e "$out" && -s "$out" ]]; then
        echo "SKIP: $out exists"
        return 0
    fi
    echo "RUN: producing $out"
    # Send command output to FD 3 (the log file), not stdout
    if bash -c "$*" >&3 2>&3; then
        echo "OK: produced $out"
    else
        echo "ERROR: command block failed"
        return 1
    fi
}

run_if_missing "$work_dir/lst-ai/space-flair_seg-lst.nii.gz" "
	lst --t1 t1.nii.gz --flair flair.nii.gz --output lst-ai --temp lst-ai/processing --probability_map
"

run_if_missing "$work_dir/lst-ai/lesion_pmap.nii.gz" "
    c3d $work_dir/lst-ai/processing/sub-X_ses-Y_space-FLAIR_seg-lst_prob.nii.gz \
    $work_dir/lst-ai/processing/sub-X_ses-Y_space-FLAIR_seg-lst_prob_1.nii.gz -max \
    $work_dir/lst-ai/processing/sub-X_ses-Y_space-FLAIR_seg-lst_prob_2.nii.gz -max \
    $work_dir/lst-ai/processing/sub-X_ses-Y_space-FLAIR_seg-lst_prob_3.nii.gz -max \
    -o $work_dir/lst-ai/lesion_pmap.nii.gz
"

exec 3>&-  # Close FD 3