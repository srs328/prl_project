base_dir=/media/smbshare/srs-9/prl_project/training/roi_train2/stage2_numcrops_dicece
for src in "$base_dir"/run*/datalist_xy20_z2.json; do
    n=$(basename "$(dirname "$src")")
    mv "$src" "$base_dir/$n/datalist_flair.phase_xy20_z2.json"
done