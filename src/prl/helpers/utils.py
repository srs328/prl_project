import numpy as np
import nibabel as nib
import os
from pathlib import Path
import pandas as pd
import pickle

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
    prl_labels = {
        
            int(prl_df.loc[subid, f"PRL{i}_label"])
            for i in range(1, 21)
            if prl_df.loc[subid, f"confidence.{i-1}"] in confidence_labels
    }

    return prl_labels


def cached(path, compute, *args, rebuild=False, **kwargs):
    path = Path(path)
    if path.exists() and not rebuild:
        with path.open("rb") as f:
            return pickle.load(f)
    result = compute(*args, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return result


def assign_prl_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    *,
    subject_col: str = "subid",
    target_col: str = "case_type",
    positive_label: str = "PRL",
    random_state: int = 42,
) -> pd.Series:
    """Return a 1-based fold ID per row, balancing PRL counts across folds."""
    if df[[subject_col, target_col]].isna().any().any():
        raise ValueError("Subject IDs and case labels must be nonmissing.")

    subjects = (
        df[target_col].eq(positive_label)
        .groupby(df[subject_col], observed=True, sort=False)
        .agg(n_prl="sum", n_lesions="size")
    )
    if not 2 <= n_splits <= len(subjects):
        raise ValueError(f"n_splits must be between 2 and {len(subjects)}.")

    # Randomize ties reproducibly, then place the hardest subjects first.
    subjects = subjects.sample(frac=1, random_state=random_state).sort_values(
        ["n_prl", "n_lesions"], ascending=False
    )

    # Each tuple tracks (PRLs, lesions, subjects) in that fold.
    loads = [(0, 0, 0) for _ in range(n_splits)]
    subject_fold = {}

    for subject, row in subjects.iterrows():
        fold = min(range(n_splits), key=loads.__getitem__)
        n_prl, n_lesions, n_subjects = loads[fold]
        loads[fold] = (n_prl + row.n_prl, n_lesions + row.n_lesions, n_subjects + 1)
        subject_fold[subject] = fold + 1

    return df[subject_col].map(subject_fold).astype(int).rename("fold")