#%%

import nibabel as nib
from pathlib import Path

# %%
subjects = []
with open("subjects.txt", 'r') as f:
    subjects = [line.strip() for line in f.readlines()]

dataroot = Path("/media/smbshare/srs-9/prl_project/training/test_train3")
for sub in subjects:
    file = dataroot / "labelsTr" / f"sub{sub}.nii.gz"
    im = nib.load(file)
    im_arr = im.get_fdata()
    im_arr[im_arr==4] = 1
    im_arr[im_arr==5] = 1
    new_im = nib.Nifti1Image(im_arr, im.affine)
    out_file = dataroot / "labelsTr" / f"sub{sub}_2.nii.gz"
    nib.save(new_im, out_file)
