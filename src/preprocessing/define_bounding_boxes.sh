#!/bin/bash

# ─── Usage ────────────────────────────────────────────────────────────────────
usage() {
    printf 'Usage: %s [-h|--help] [--expand-xy <arg>] [--expand-z <arg>] work_dir \n' "$0"
    printf '\n'
    printf 'Positional arguments:\n'
    printf '    work_dir              Required working direectory\n'
    printf '\n'
    printf 'Options:\n'
    printf '  -h, --help            Show this help message and exit\n'
    printf '  --expand-xy           How much to expand in xy plane (default: 20)\n'
    printf '  --expand-z            How much to expand in z axis (default: 2)\n'
}

expand_xy=20
expand_z=2

while [[ $# -gt 0 ]]; do
    case "$1" in

        --expand-xy)
            [[ -z "${2-}" ]] && { echo "Error: --expand-xy requires an argument"; exit 1; }
            expand_xy="$2"
            shift 2
            ;;
        --expand-z)
            [[ -z "${2-}" ]] && { echo "Error: --expand-z requires an argument"; exit 1; }
            expand_z="$2"
            shift 2
            ;;
        # Unknown flags (anything starting with -)
        -*) 
            echo "Error: Unknown option: $1"; usage; exit 1
            ;;
        # Positional arguments (anything not starting with -)
        *)
            positional_args+=("$1")
            shift
            ;;
    esac
done
if [[ ${#positional_args[@]} -eq 0 ]]; then
    echo "Error: a work_dir is required"
    usage; exit 1
fi
work_dir="${positional_args[0]}"

echo "Working in ${work_dir}"
cd "$work_dir" || exit 1


bounding_boxes=lstai_bounding_boxes.txt

# Only generate if it doesn't exist
if [ ! -f "$bounding_boxes" ]; then
    n_lesions=$(fslstats lstai_lesion_index.nii.gz -R | awk '{printf "%d\n", $2}')
    echo "Found ${n_lesions} lesions"
    > "$bounding_boxes"   # create empty
    for lesion_label in $(seq 1 "$n_lesions"); do
        temp_BBox=$(fslstats lstai_lesion_index.nii.gz \
            -l $(bc <<< "${lesion_label} - 0.5") \
            -u $(bc <<< "${lesion_label} + 0.5") \
            -w)
        echo "$lesion_label $temp_BBox" >> "$bounding_boxes"
    done
else
    echo "Using cached $bounding_boxes"
fi

echo "will expand xy dimension of PRL by $expand_xy"
expanded_boxes="lstai_bounding_boxes_xy${expand_xy}_z${expand_z}.txt"
while read -r lesion_label bbox_rest; do
    roi_boundaries=$(echo "$bbox_rest" | awk -v expand_xy="$expand_xy" -v expand_z="$expand_z" '{
        printf "%d %d %d %d %d %d\n", $1-expand_xy, $2+2*expand_xy, $3-expand_xy, $4+2*expand_xy, $5-expand_z, $6+2*expand_z
    }')
    echo "$lesion_label $roi_boundaries" >> "$expanded_boxes"
done < "$bounding_boxes"

echo "ROI bounding boxes saved to ${expanded_boxes}"
