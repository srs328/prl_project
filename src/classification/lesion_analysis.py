"""Lesion-level analysis of reference labels and model predictions."""

from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger
from scipy import ndimage

from .geometry import (
    compute_pca_features,
    compute_radial_metrics,
    get_convex_hull,
    rim_enclosing_sphere_radius,
)

__all__ = [
    "analyze_label",
    "analyze_prl_case",
    "analyze_subject_prl",
    "analyze_subject_prl0",
    "compute_pca_features",
    "compute_radial_metrics",
    "count_predicted_prls",
    "count_rim_for_lesion",
    "crop_from_volume",
    "get_center_label",
    "get_center_lesion",
    "get_convex_hull",
    "get_lesion_rim",
    "get_rim_units",
    "parse_bounding_boxes",
    "rim_enclosing_sphere_radius",
    "screen_for_iron",
]


def parse_bounding_boxes(bbox_file: Path) -> list[tuple[int, list[int]]]:
    """Parse bounding box file into list of (index, [xmin, xsize, ymin, ysize, zmin, zsize])."""
    bounding_boxes = []
    with open(bbox_file) as f:
        for line in f:
            parts = line.split()
            index = int(parts[0])
            coords = list(map(int, parts[1:]))
            bounding_boxes.append((index, coords))
    return bounding_boxes


def crop_from_volume(volume: np.ndarray, coords: list[int]) -> np.ndarray:
    """Extract a crop from a full-brain volume using fslroi-style bounding box coords.

    Handles negative start coords (from bbox expansion beyond brain boundary)
    by zero-padding, matching fslroi's behavior.

    Args:
        volume: Full-brain 3D array.
        coords: [xmin, xsize, ymin, ysize, zmin, zsize].

    Returns:
        Cropped array of shape (xsize, ysize, zsize).
    """
    xmin, xsize, ymin, ysize, zmin, zsize = coords
    brain_shape = volume.shape[:3]

    crop = np.zeros((xsize, ysize, zsize), dtype=volume.dtype)

    slices_brain = []
    slices_crop = []
    for start, size, brain_dim in zip(
        [xmin, ymin, zmin], [xsize, ysize, zsize], brain_shape
    ):
        b_start = max(0, start)
        b_end = min(brain_dim, start + size)
        c_start = max(0, -start)
        c_end = c_start + (b_end - b_start)

        if b_end <= b_start:
            return crop

        slices_brain.append(slice(b_start, b_end))
        slices_crop.append(slice(c_start, c_end))

    crop[tuple(slices_crop)] = volume[tuple(slices_brain)]
    return crop


def get_center_lesion(
    index_crop: np.ndarray, label_data: np.ndarray, lesion_id: int
) -> np.ndarray:
    """Get boolean mask of the center lesion's body (label=1 voxels).

    Uses connected components to find lesion regions that overlap with
    the lesion index mask, excluding distant neighbor lesions.

    Args:
        index_crop: Cropped lstai_lesion_index (same shape as label_data).
        label_data: Segmentation output (0=bg, 1=lesion, 2=rim).
        lesion_id: The central lesion's integer ID.

    Returns:
        Boolean mask of the center lesion.
    """
    return get_center_label(index_crop, label_data, lesion_id, label_class=1)


def get_center_label(
    index_crop: np.ndarray,
    label_data: np.ndarray,
    lesion_id: int,
    label_class: int,
    n_dilate: int | None = None,
) -> np.ndarray:
    """Get boolean mask of a specific label class for a specific lesion.

    Uses connected components to find label_class regions that overlap with the
    central lesion's footprint (optionally dilated).

    Args:
        index_crop: Cropped lstai_lesion_index.
        label_data: Segmentation output (0=bg, 1=lesion, 2=rim).
        lesion_id: The central lesion's integer ID.
        label_class: Label value to extract (1=lesion, 2=rim).
        n_dilate: Voxels to dilate the lesion footprint by.

    Returns:
        Boolean mask of the requested label class for this lesion.
    """
    labeled, _ = ndimage.label(label_data == label_class)
    index_mask = index_crop == lesion_id
    if n_dilate is not None:
        index_mask = ndimage.binary_dilation(index_mask, iterations=n_dilate)

    overlapping = np.unique(labeled[index_mask])
    return np.isin(labeled, overlapping[overlapping != 0])


def get_lesion_rim(
    index_crop: np.ndarray,
    label_data: np.ndarray,
    lesion_id: int,
    n_dilate: int = 1,
) -> np.ndarray:
    """Get boolean mask of rim voxels belonging to a specific lesion.

    Uses connected components to find label=2 regions that overlap with the
    central lesion's footprint (dilated to capture rim just outside boundary).

    Args:
        index_crop: Cropped region from lstai_lesion_index.nii.gz.
        label_data: Segmentation output (0=bg, 1=lesion, 2=rim).
        lesion_id: The central lesion's integer ID.
        n_dilate: Voxels to dilate the lesion footprint by.

    Returns:
        Boolean mask of rim voxels for this lesion.
    """
    return get_center_label(
        index_crop, label_data, lesion_id, label_class=2, n_dilate=n_dilate
    )


def get_rim_units(rim):
    concomp_s = ndimage.generate_binary_structure(3, 3)
    rim_close1 = ndimage.binary_closing(rim, iterations=2, structure=concomp_s)
    labeled, n_components = ndimage.label(rim_close1, structure=concomp_s)
    sorted_comps = sorted(
        range(1, n_components + 1),
        key=lambda comp_id: np.sum(labeled == comp_id),
        reverse=True,
    )

    # just do simple thresholding for now
    components = [comp_i for comp_i in sorted_comps if np.sum(labeled == comp_i) > 10]
    for comp_i in components:
        print(f"Component {comp_i} size: ", np.sum(labeled == comp_i), " voxels")
    labeled[~np.isin(labeled, components)] = 0

    # Now do a more careful closure
    # ? What would using generate_binary_structure(3,1) change practically? Is there an
    # ?   obviously better choice if I want a 3D shell-like shape?
    concomp_s = ndimage.generate_binary_structure(3, 1)
    rim_close2 = ndimage.binary_closing(rim, iterations=2, structure=concomp_s)

    return rim_close2 * labeled, components


def count_rim_for_lesion(
    index_crop: np.ndarray,
    label_data: np.ndarray,
    lesion_id: int,
    n_dilate: int = 1,
) -> int:
    """Count rim voxels belonging to a specific lesion."""
    return int(get_lesion_rim(index_crop, label_data, lesion_id, n_dilate).sum())


def _load_subject_geometry(subject):
    index_volume = np.asanyarray(subject.load_nifti("lstai_lesion_index").dataobj)
    return index_volume.astype(np.int32, copy=False), dict(subject.bounding_boxes)


def _label_path(value, dataset) -> Path | None:
    if pd.isna(value):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = dataset.data_root / path
    if not path.is_file():
        logger.warning(f"Label file not found: {path}")
        return None
    return path


def _has_iron(index_crop, label_data, lesion_index) -> bool:
    if index_crop.shape != label_data.shape:
        raise ValueError(
            f"Label shape {label_data.shape} differs from index crop shape {index_crop.shape}"
        )
    return bool(np.any((index_crop == lesion_index) & (label_data == 2)))


def screen_for_iron(dataset, subid, label_key) -> dict[int, bool]:
    """Flag lesions with label-2 voxels inside their indexed footprint.

    An unavailable label file receives False so the result can be used as a
    boolean filter.
    """
    index_volume, boxes = _load_subject_geometry(dataset.subject(subid))
    has_iron = {}
    for lesion_index, case in dataset.cases.loc[subid].iterrows():
        path = _label_path(case.get(label_key), dataset)
        if path is None:
            has_iron[lesion_index] = False
            continue

        label = np.asanyarray(nib.load(path).dataobj)
        has_iron[lesion_index] = bool(np.any(label == 2)) and _has_iron(
            crop_from_volume(index_volume, boxes[lesion_index]), label, lesion_index
        )
    return has_iron


def _measure_label(
    index_crop, label_data, lesion_index, voxel_size, *,
    include_data, key, screen_iron, shape_features,
):
    key = key.removeprefix("_")
    has_iron = _has_iron(index_crop, label_data, lesion_index)
    stats = {f"has_iron_{key}": has_iron}
    data = {f"index_crop_{key}": index_crop} if include_data else {}
    if screen_iron and not has_iron:
        return stats, data

    voxel_volume = math.prod(voxel_size)
    rim = get_lesion_rim(index_crop, label_data, lesion_index)
    lesion = get_center_lesion(index_crop, label_data, lesion_index)

    for name, mask in (("rim", rim), ("lesion", lesion)):
        count = int(mask.sum())
        hull = get_convex_hull(mask, voxel_sizes=voxel_size)
        stats[f"{name}_voxels_{key}"] = count
        stats[f"{name}_volume_{key}"] = count * voxel_volume
        if hull is not None:
            stats[f"{name}_hull_volume_{key}"] = hull.volume
        if include_data:
            data[f"{name}_{key}"] = mask
            data[f"{name}_hull_{key}"] = hull

    stats[f"rim_sphere_radius_{key}"] = rim_enclosing_sphere_radius(rim, voxel_size)
    if include_data:
        data[f"voxel_size_{key}"] = voxel_size

    if shape_features and key == "infer":
        stats.update(compute_pca_features(rim))
        stats.update(compute_radial_metrics(rim))
    return stats, data


def analyze_label(
    index_crop: np.ndarray,
    lab_nifti: nib.Nifti1Image,
    lesion_index: int,
    include_data: bool = True,
    key: str = "infer",
    screen_iron: bool = True,
) -> tuple[dict, dict]:
    """Return scalar measurements and optional arrays for a cropped label.

    With screening enabled, the strict label-2 overlap test gates the geometry
    measurements. Rim extraction itself uses a dilated footprint. Inference
    labels also receive PCA and radial features, including in stats-only mode.
    """
    return _measure_label(
        index_crop,
        np.asanyarray(lab_nifti.dataobj),
        lesion_index,
        lab_nifti.header.get_zooms()[:3],
        include_data=include_data,
        key=key,
        screen_iron=screen_iron,
        shape_features=True,
    )


def _analyze_subject_prl(
    subid, dataset, label_keys, lesion_indices, include_data, cases, *,
    screen_iron, legacy=False,
) -> tuple[list[dict], list[dict]]:
    if isinstance(label_keys, str):
        label_keys = (label_keys,)
    else:
        label_keys = tuple(label_keys)
    if cases is None:
        cases = dataset.cases
    elif not isinstance(cases, pd.DataFrame):
        logger.warning("Expected a DataFrame for cases. Using dataset.cases.")
        cases = dataset.cases
    subject_cases = cases.loc[subid]
    if lesion_indices is None:
        lesion_indices = subject_cases.index

    index_volume, boxes = _load_subject_geometry(dataset.subject(subid))
    subject_stats = []
    subject_data = []
    label_suffixes = {} if legacy else {"inference": "infer", "label": "truth"}
    for lesion_index in lesion_indices:
        case = subject_cases.loc[lesion_index]
        identifiers = {"subid": subid, "lesion_index": lesion_index}
        if not legacy:
            identifiers["case_type"] = case.get("case_type")
        lesion_stats = identifiers.copy()
        lesion_data = identifiers.copy()
        index_crop = crop_from_volume(index_volume, boxes[lesion_index])
        for key in label_keys:
            path = _label_path(case.get(key), dataset)
            if path is None:
                continue
            label_key = label_suffixes.get(key, key)
            try:
                nifti = nib.load(path)
                stats, data = _measure_label(
                    index_crop,
                    np.asanyarray(nifti.dataobj),
                    lesion_index,
                    nifti.header.get_zooms()[:3],
                    include_data=include_data,
                    key=label_key,
                    screen_iron=screen_iron,
                    shape_features=not legacy,
                )
            except (OSError, ValueError, nib.filebasedimages.ImageFileError):
                logger.opt(exception=True).warning(
                    f"Label analysis failed for sub{subid} lesion {lesion_index} ({key})"
                )
                continue
            lesion_stats.update(stats)
            lesion_data.update(data)

        subject_stats.append(lesion_stats)
        subject_data.append(lesion_data)
    return subject_stats, subject_data


def analyze_subject_prl(
    subid,
    dataset,
    label_keys,
    lesion_indices=None,
    include_data=True,
    cases: pd.DataFrame | None = None,
    *,
    screen_iron: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Analyze selected lesions using one load of the subject's index and boxes.

    The inference and label columns use the "_infer" and "_truth" suffixes.
    Other label columns keep their names. Missing files leave results from the
    other labels intact.
    """
    return _analyze_subject_prl(
        subid, dataset, label_keys, lesion_indices, include_data, cases,
        screen_iron=screen_iron,
    )


def analyze_subject_prl0(
    dataset, subid, label_keys, include_data=True, cases: pd.DataFrame | None = None
) -> tuple[list[dict], list[dict]]:
    """Original subject analysis with unscreened geometry and column-name suffixes."""
    return _analyze_subject_prl(
        subid, dataset, label_keys, None, include_data, cases,
        screen_iron=False, legacy=True,
    )


def analyze_prl_case(
    case,
    dataset,
    include_data=True,
    screen_iron=False,
    count_rim=False,
):
    """Analyze a single case from a Dataset or experiment.

    Returns statistics and optional label arrays for the available truth and
    inference files. Returns None when both files are unavailable. With
    screen_iron=True, returns strict rim overlap for the first available file
    in truth-then-inference order. count_rim is accepted for older callers.
    """
    subid, lesion_index = (
        case.name if isinstance(case, pd.Series) else (case["subid"], case["lesion_index"])
    )
    label_paths = {}
    for column, key in (("label", "truth"), ("inference", "infer")):
        path = _label_path(case.get(column), dataset)
        if path is not None:
            label_paths[key] = path
    if not label_paths:
        logger.warning(f"No label or inference found for sub{subid} lesion {lesion_index}")
        return None

    index_volume, boxes = _load_subject_geometry(dataset.subject(subid))
    index_crop = crop_from_volume(index_volume, boxes[lesion_index])
    lesion_stats = {"subid": subid, "lesion_index": lesion_index}
    lesion_data = lesion_stats.copy()
    for key, path in label_paths.items():
        nifti = nib.load(path)
        label = np.asanyarray(nifti.dataobj).astype(np.uint8, copy=False)
        if screen_iron:
            return _has_iron(index_crop, label, lesion_index)
        if include_data:
            lesion_data[f"label_{key}"] = label
        try:
            stats, data = _measure_label(
                index_crop,
                label,
                lesion_index,
                nifti.header.get_zooms()[:3],
                include_data=include_data,
                key=key,
                screen_iron=False,
                shape_features=False,
            )
        except ValueError:
            logger.opt(exception=True).warning(
                f"Label analysis failed for sub{subid} lesion {lesion_index} ({key})"
            )
            continue
        if key == "truth":
            stats.pop("has_iron_truth")
        lesion_stats.update(stats)
        lesion_data.update(data)
    return lesion_stats, lesion_data


def count_predicted_prls(dataset, subid, *, cases: pd.DataFrame | None = None) -> dict:
    """Summarize PRLs for a subject using a case table with inference paths.

    Pass an experiment's case table to count predictions from that run. Lesions
    count as PRLs when predicted label-2 voxels overlap the indexed footprint.
    Total lesions comes from the subject's bounding boxes.
    """
    if cases is None:
        cases = dataset.cases
    if "inference" not in cases:
        raise ValueError("Expected an 'inference' column with prediction paths.")
    subject = dataset.subject(subid)
    index_volume, boxes = _load_subject_geometry(subject)
    prls = []
    for lesion_index, case in cases.loc[subid].iterrows():
        path = _label_path(case.get("inference"), dataset)
        if path is None:
            continue
        nifti = nib.load(path)
        label = np.asanyarray(nifti.dataobj)
        index_crop = crop_from_volume(index_volume, boxes[lesion_index])
        if not _has_iron(index_crop, label, lesion_index):
            continue

        rim = get_lesion_rim(index_crop, label, lesion_index)
        voxel_size = nifti.header.get_zooms()[:3]
        hull = get_convex_hull(rim, voxel_sizes=voxel_size)
        prls.append({
            "lesion_index": lesion_index,
            "rim_voxels": int(rim.sum()),
            "hull_volume_mm3": hull.volume if hull is not None else None,
            "sphere_radius_mm": rim_enclosing_sphere_radius(rim, voxel_size),
        })

    return {
        "subject": subject.dir.name,
        "total_lesions": len(boxes),
        "predicted_prls": len(prls),
        "prls": prls,
    }
