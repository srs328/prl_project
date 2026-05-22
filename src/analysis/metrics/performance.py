"""Performance metrics: confusion matrix, derived metrics, casewise stats.

All performance/confusion logic lives here — get_confusion_matrix through
performance_metrics() — since they're part of the same computation chain.

Functions:
    get_confusion_matrix: TP/FP/TN/FN from label + inference NIfTIs.
    compute_derived_metrics: Sensitivity, specificity, F1, etc. from confusion counts.
    compute_casewise_stats: Per-case metrics from a cases DataFrame.
    performance_metrics: Aggregate cached case_performance into a single row dict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.spatialimages import SpatialImage
from loguru import logger

from helpers.utils import dice_score


def get_confusion_matrix(
    lab: SpatialImage, inf: SpatialImage
) -> tuple[int, int, int, int]:
    """Compute confusion matrix from label and inference images.

    Treats label 1 (lesion) as negative, label 2 (rim) as positive.

    Returns:
        (TP, FP, TN, FN)
    """
    lab_data = lab.get_fdata()
    inf_data = inf.get_fdata()
    try:
        TP = np.sum((lab_data == 2) & (inf_data == 2))
        FP = np.sum((lab_data == 1) & (inf_data == 2))
        TN = np.sum((lab_data == 1) & (inf_data == 1))
        FN = np.sum((lab_data == 2) & (inf_data == 1))
        return int(TP), int(FP), int(TN), int(FN)
    except Exception as e:
        logger.warning(f"Failed to compute confusion matrix: {e}")
        return 0, 0, 0, 0


def compute_derived_metrics(
    tp: int, fp: int, tn: int, fn: int
) -> dict[str, float]:
    """Compute derived metrics from confusion matrix counts.

    Returns dict with: sensitivity, specificity, fpr, fnr, precision,
    npv, accuracy, f1.
    """
    metrics = {}

    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    metrics["fpr"] = fp / (tn + fp) if (tn + fp) > 0 else np.nan
    metrics["fnr"] = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    total = tp + fp + tn + fn
    metrics["accuracy"] = (tp + tn) / total if total > 0 else np.nan

    precision = metrics["precision"]
    recall = metrics["sensitivity"]
    if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
        metrics["f1"] = 2 * (precision * recall) / (precision + recall)
    else:
        metrics["f1"] = np.nan

    return metrics


def compute_casewise_stats(cases) -> list[dict]:
    """Compute per-case performance stats from a cases DataFrame or list of dicts.

    Each case must have 'label', 'inference', 'subid', 'lesion_index',
    'split', and 'case_type' fields/columns.

    Args:
        cases: DataFrame (from Dataset.cases or Experiment.cases) or list of dicts.
            Only cases where inference is not None are processed.

    Returns:
        List of dicts, one per case with confusion matrix + derived metrics.
    """
    import pandas as pd

    if isinstance(cases, pd.DataFrame):
        rows = []
        for idx, row in cases.iterrows():
            if row.get("inference") is None:
                continue
            subid, lesion_index = idx if isinstance(idx, tuple) else (row.get("subid"), row.get("lesion_index"))
            rows.append({
                "subid": subid,
                "lesion_index": lesion_index,
                "split": row["split"],
                "case_type": row["case_type"],
                "label": row["label"],
                "inference": row["inference"],
            })
    else:
        rows = [item for item in cases if item.get("inference") is not None]

    results = []
    for item in rows:
        lab_path = Path(item["label"])
        inf_path = Path(item["inference"])

        lab = nib.load(lab_path)
        inf = nib.load(inf_path)
        prl_dice = dice_score(lab.get_fdata(), inf.get_fdata(), seg1_val=2, seg2_val=2)
        lesion_dice = dice_score(lab.get_fdata(), inf.get_fdata(), seg1_val=1, seg2_val=1)
        tp, fp, tn, fn = get_confusion_matrix(lab, inf)

        metrics = {
            "subid": item["subid"],
            "lesion_index": item["lesion_index"],
            "split": item["split"],
            "case_type": item["case_type"],
            "lesion_dice": lesion_dice,
            "prl_dice": prl_dice,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        metrics.update(compute_derived_metrics(tp, fp, tn, fn))

        if (tp + fp + tn + fn) == 0:
            metrics["Notes"] = "Something's wrong"
        else:
            metrics["Notes"] = None

        results.append(metrics)

    return results


def compute_metrics(lab_path, inf_path):
    lab = nib.load(lab_path)
    inf = nib.load(inf_path)
    prl_dice = dice_score(lab.get_fdata(), inf.get_fdata(), seg1_val=2, seg2_val=2)
    lesion_dice = dice_score(lab.get_fdata(), inf.get_fdata(), seg1_val=1, seg2_val=1)
    tp, fp, tn, fn = get_confusion_matrix(lab, inf)

    metrics = {
        "lesion_dice": lesion_dice,
        "prl_dice": prl_dice,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    metrics.update(compute_derived_metrics(tp, fp, tn, fn))

    if (tp + fp + tn + fn) == 0:
        metrics["Notes"] = "Something's wrong"
    else:
        metrics["Notes"] = None
    return metrics


import numpy as np
from scipy import ndimage


def _surface_distances(mask_a, mask_b, voxel_spacing=None):
    """
    Compute distances from the surface of mask_a to the surface of mask_b.

    Returns an array of distances. If either mask is empty, returns None.
    """
    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)

    if not mask_a.any() or not mask_b.any():
        return None

    if voxel_spacing is None:
        voxel_spacing = np.ones(mask_a.ndim)

    # Define surfaces as mask voxels minus interior voxels
    structure = ndimage.generate_binary_structure(mask_a.ndim, 1)

    surface_a = mask_a ^ ndimage.binary_erosion(mask_a, structure=structure, border_value=0)
    surface_b = mask_b ^ ndimage.binary_erosion(mask_b, structure=structure, border_value=0)

    # Distance transform gives distance to nearest surface_b voxel
    dt_b = ndimage.distance_transform_edt(~surface_b, sampling=voxel_spacing)

    return dt_b[surface_a]


def _hausdorff_metrics(mask_a, mask_b, voxel_spacing=None, percentile=95, zero_division=np.nan):
    """
    Symmetric Hausdorff and percentile Hausdorff distance between two binary masks.

    Returns:
        hausdorff: maximum symmetric surface distance
        hausdorff95: percentile symmetric surface distance
    """
    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)

    if not mask_a.any() and not mask_b.any():
        return zero_division, zero_division

    if not mask_a.any() or not mask_b.any():
        return np.inf, np.inf

    dists_a_to_b = _surface_distances(mask_a, mask_b, voxel_spacing)
    dists_b_to_a = _surface_distances(mask_b, mask_a, voxel_spacing)

    if dists_a_to_b is None or dists_b_to_a is None:
        return np.inf, np.inf

    all_dists = np.concatenate([dists_a_to_b, dists_b_to_a])

    hausdorff = np.max(all_dists)
    hausdorff95 = np.percentile(all_dists, percentile)

    return hausdorff, hausdorff95


def binary_segmentation_metrics(
    lab_data,
    inf_data,
    positive_label=2,
    roi_mask=None,
    voxel_spacing=None,
    zero_division=np.nan,
):
    """
    Compute binary segmentation metrics for a target label.

    By default, evaluates positive_label vs all other voxels.
    If roi_mask is supplied, metrics are computed only within that mask.

    Parameters
    ----------
    lab_data : array-like
        Ground truth label image.

    inf_data : array-like
        Inferred/predicted label image.

    positive_label : int
        Label treated as the positive class.

    roi_mask : array-like or None
        Optional mask restricting evaluation to a region of interest.

    voxel_spacing : tuple/list/array or None
        Physical voxel spacing, e.g. (1.0, 1.0, 1.5). If None, Hausdorff
        distances are reported in voxel units.

    zero_division : float
        Value returned when a metric denominator is zero.

    Returns
    -------
    dict
        Dice/F1, confusion counts, precision, recall, specificity, NPV,
        accuracy, Hausdorff distance, and 95th percentile Hausdorff distance.
    """

    lab_data = np.asarray(lab_data)
    inf_data = np.asarray(inf_data)

    if roi_mask is None:
        roi_mask = np.ones(lab_data.shape, dtype=bool)
    else:
        roi_mask = np.asarray(roi_mask).astype(bool)

    manual_pos = (lab_data == positive_label) & roi_mask
    pred_pos = (inf_data == positive_label) & roi_mask

    manual_neg = (~manual_pos) & roi_mask
    pred_neg = (~pred_pos) & roi_mask

    TP = np.sum(manual_pos & pred_pos)
    FP = np.sum(manual_neg & pred_pos)
    TN = np.sum(manual_neg & pred_neg)
    FN = np.sum(manual_pos & pred_neg)

    def safe_div(num, den):
        return num / den if den != 0 else zero_division

    dice = safe_div(2 * TP, 2 * TP + FP + FN)
    precision = safe_div(TP, TP + FP)
    sensitivity = safe_div(TP, TP + FN)
    recall = sensitivity
    specificity = safe_div(TN, TN + FP)
    npv = safe_div(TN, TN + FN)
    accuracy = safe_div(TP + TN, TP + FP + TN + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)

    hausdorff, hausdorff95 = _hausdorff_metrics(
        manual_pos,
        pred_pos,
        voxel_spacing=voxel_spacing,
        percentile=95,
        zero_division=zero_division,
    )

    return {
        "dice": dice,
        "f1": f1,
        "tp": int(TP),
        "fp": int(FP),
        "tn": int(TN),
        "fn": int(FN),
        "precision": precision,
        "sensitivity": sensitivity,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "accuracy": accuracy,
        "hausdorff": hausdorff,
        "hausdorff95": hausdorff95,
    }
    

def performance_metrics(
    experiment_data: dict,
    splits: list[str] | None = None,
) -> dict:
    """Aggregate cached case_performance into a single row of performance metrics.

    Uses pre-computed per-case stats from experiment_data['case_performance']
    (produced by compute_casewise_stats), so no NIfTI reloading needed.

    Args:
        experiment_data: Dict from load_run_data / cached_load_run_data.
        splits: Which splits to include (e.g. ["testing"], ["fold0", "fold1"]).
            Defaults to ["testing"].

    Returns:
        Flat dict suitable for one DataFrame row.
    """
    if splits is None:
        splits = ["testing"]

    case_performance = experiment_data.get("case_performance", [])
    if not case_performance:
        return {"perf_status": "no_cases"}

    cases = [c for c in case_performance if c.get("split") in splits]
    if not cases:
        return {"perf_status": f"no_cases_in_{','.join(splits)}"}

    total_tp = total_fp = total_tn = total_fn = 0
    valid_cases = 0
    rim_case_count = 0
    total_rim_voxels = 0
    total_lesion_voxels = 0
    lesion_case_count = 0

    for c in cases:
        tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
        if (tp + fp + tn + fn) == 0:
            continue
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        total_rim_voxels += tp + fn
        total_lesion_voxels += tn + fp
        valid_cases += 1
        if c["case_type"] == "PRL":
            rim_case_count += 1
        else:
            lesion_case_count += 1

    row = {
        "perf_status": "complete",
        "perf_case_count": valid_cases,
        "rim_case_count": rim_case_count,
        "n_rim_voxels": total_rim_voxels,
        "lesion_case_count": lesion_case_count,
        "n_lesion_voxels": total_lesion_voxels,
    }

    if valid_cases == 0:
        row["perf_status"] = "no_valid_cases"
        return row

    agg = compute_derived_metrics(total_tp, total_fp, total_tn, total_fn)
    for k, v in agg.items():
        row[f"perf_{k}"] = np.round(v, 4) if not np.isnan(v) else np.nan

    row["perf_tp"] = total_tp
    row["perf_fp"] = total_fp
    row["perf_tn"] = total_tn
    row["perf_fn"] = total_fn

    # Per-case summary stats (PRL cases only -- Lesion cases have NaN sensitivity)
    prl_cases = [
        c
        for c in cases
        if c.get("case_type") == "PRL" and (c["tp"] + c["fp"] + c["tn"] + c["fn"]) > 0
    ]
    for metric in [
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "prl_dice",
        "lesion_dice",
    ]:
        values = [c[metric] for c in prl_cases if not np.isnan(c.get(metric, np.nan))]
        if values:
            row[f"perf_{metric}_mean"] = np.round(np.mean(values), 4)
            row[f"perf_{metric}_std"] = np.round(np.std(values), 4)

    return row
