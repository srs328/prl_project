import numpy as np
import nibabel as nib
import os

def dice_score(seg1, seg2, seg1_val=1, seg2_val=1):
    if isinstance(seg1, str) or isinstance(seg1, os.PathLike):
        seg1 = nib.load(seg1).get_fdata()
        seg2 = nib.load(seg2).get_fdata()
    intersection = np.sum((seg1 == seg1_val) & (seg2 == seg2_val))
    volume_sum = np.sum(seg1 == seg1_val) + np.sum(seg2 == seg2_val)
    if volume_sum == 0:
        # ? Why did I originally make this 1.0? Was there good reason, or mistake?
        # return 1.0
        return None
    return 2.0 * intersection / volume_sum
