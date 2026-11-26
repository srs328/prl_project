import numpy as np
import nibabel as nib
import os

def dice_score(seg1: os.PathLike | np.ndarray, seg2: os.PathLike | np.ndarray,
               seg1_val: int = 1, seg2_val: int = 1):
    """Compute Dice coefficient between two segmentations for specified label indices"""
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


def get_prl_indices(prl_df, subid, possible=False):
    confidence_labels = ["definite", "probable"]
    if possible:
        confidence_labels.append("possible")
    prl_labels = set(
        [
            int(prl_df.loc[subid, f"PRL{i}_label"])
            for i in range(1, 21)
            if prl_df.loc[subid, f"confidence.{i-1}"] in confidence_labels
        ]
    )
    return prl_labels