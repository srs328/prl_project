"""Lesion-level analysis: rim extraction, PRL counting, case analysis.

Functions:
    parse_bounding_boxes: Parse bbox file into (index, coords) pairs.
    crop_from_volume: Extract a crop matching fslroi bounding box coords.
    get_center_lesion: Boolean mask of the center lesion's body.
    get_center_label: Boolean mask of a label class for a specific lesion.
    get_lesion_rim: Boolean mask of rim voxels for a specific lesion.
    count_rim_for_lesion: Count rim voxels for a specific lesion.
    analyze_prl_case: Full analysis of a single PRL case.
"""

from __future__ import annotations
import os
from collections.abc import Iterable
import math
import pandas as pd
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from loguru import logger

from analysis.image.geometry import (
    get_convex_hull,
    rim_enclosing_sphere_radius,
    compute_radial_metrics,
    compute_pca_features
)


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
    lesion_mask = label_data == 1
    labeled, n_components = ndimage.label(lesion_mask)
    index_mask = index_crop == lesion_id

    result = np.zeros_like(lesion_mask)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if np.any(comp_mask & index_mask):
            result |= comp_mask

    return result


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
    label_mask = label_data == label_class
    labeled, n_components = ndimage.label(label_mask)

    index_mask = index_crop == lesion_id
    if n_dilate is not None:
        index_mask = ndimage.binary_dilation(index_mask, iterations=n_dilate)

    result = np.zeros_like(label_mask)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if np.any(comp_mask & index_mask):
            result |= comp_mask

    return result


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
        index_crop: Cropped lstai_lesion_index.
        label_data: Segmentation output (0=bg, 1=lesion, 2=rim).
        lesion_id: The central lesion's integer ID.
        n_dilate: Voxels to dilate the lesion footprint by.

    Returns:
        Boolean mask of rim voxels for this lesion.
    """
    rim_mask = label_data == 2

    labeled, n_components = ndimage.label(rim_mask)

    index_mask = index_crop == lesion_id
    dilated = ndimage.binary_dilation(index_mask, iterations=n_dilate)

    result = np.zeros_like(rim_mask)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if np.any(comp_mask & dilated):
            result |= comp_mask

    return result

def get_rim_units(rim):
    concomp_s = ndimage.generate_binary_structure(3,3)
    rim_close1 = ndimage.binary_closing(rim, iterations=2, structure=concomp_s)
    labeled, n_components = ndimage.label(rim_close1, structure=concomp_s)
    sorted_comps = sorted(range(1, n_components+1), key=lambda comp_id: np.sum(labeled==comp_id), reverse=True)

    # just do simple thresholding for now
    components = [comp_i for comp_i in sorted_comps if np.sum(labeled==comp_i) > 10]
    for comp_i in components:
        print(f"Component {comp_i} size: ", np.sum(labeled==comp_i), " voxels")
    labeled[~np.isin(labeled, components)] = 0
        
    # Now do a more careful closure
    #? What would using generate_binary_structure(3,1) change practically? Is there an 
    #?   obviously better choice if I want a 3D shell-like shape?
    concomp_s = ndimage.generate_binary_structure(3,1)
    rim_close2 = ndimage.binary_closing(rim, iterations=2, structure=concomp_s)

    return rim_close2*labeled, components



def count_rim_for_lesion(
    index_crop: np.ndarray,
    label_data: np.ndarray,
    lesion_id: int,
    n_dilate: int = 1,
) -> int:
    """Count rim voxels belonging to a specific lesion."""
    return int(get_lesion_rim(index_crop, label_data, lesion_id, n_dilate).sum())


def screen_for_iron(dataset, subid, label_key):
    subject = dataset.subject(subid)
    lesion_index_path = subject.dir / "lstai_lesion_index.nii.gz"
    lesion_index_vol = nib.load(str(lesion_index_path)).get_fdata().astype(np.int32)
    # Parse bounding boxes
    cfg = dataset.preprocess
    bbox_suffix = f"xy{cfg.expand_xy}_z{cfg.expand_z}"
    bbox_file = subject.dir / f"lstai_bounding_boxes_{bbox_suffix}.txt"
    bounding_boxes = parse_bounding_boxes(bbox_file)

    has_iron = {}
    for lesion_index, case in dataset.cases.loc[subid, :].iterrows():
        lab_path = case.get(label_key)
        if lab_path is None:
            has_iron[lesion_index] = "Missing"
            continue
        coords = bounding_boxes[lesion_index - 1][1]
        lab_nifti = nib.load(str(lab_path))
        lab_data = np.asanyarray(lab_nifti.dataobj).astype(np.uint8)
        if not np.any(lab_data == 2):
            has_iron[lesion_index] = False
            continue

        # Only then do the more complex intersection check
        index_crop = crop_from_volume(lesion_index_vol, coords)
        has_iron[lesion_index] = np.any((index_crop == lesion_index) & (lab_data == 2))
    return has_iron


def analyze_subject_prl0(dataset, subid, label_keys, include_data=True, cases: pd.DataFrame | None = None):
    if not isinstance(label_keys, Iterable):
        label_keys = [label_keys]
    
    if cases is None:
        cases = dataset.cases
    elif not isinstance(cases, pd.DataFrame):
        print("cases argument must be None or pd.DataFrame, but defaulting to dataset.cases")
        cases = dataset.cases
    subject = dataset.subject(subid)
    lesion_index_path = subject.dir / "lstai_lesion_index.nii.gz"
    lesion_index_vol = nib.load(str(lesion_index_path)).get_fdata().astype(np.int32)
    # Parse bounding boxes
    cfg = dataset.preprocess
    bbox_suffix = f"xy{cfg.expand_xy}_z{cfg.expand_z}"
    bbox_file = subject.dir / f"lstai_bounding_boxes_{bbox_suffix}.txt"
    bounding_boxes = parse_bounding_boxes(bbox_file)

    subject_lesion_data = []
    subject_lesion_stats = []
    for lesion_index, lesion_case in cases.loc[subid, :].iterrows():
        lesion_data = {"subid": subid, "lesion_index": lesion_index} 
        lesion_stats = {"subid": subid, "lesion_index": lesion_index} 
        coords = bounding_boxes[lesion_index - 1][1]       
        index_crop = crop_from_volume(lesion_index_vol, coords)
        for key in label_keys:
            lab_path = lesion_case.get(key)
            if lab_path is None or not os.path.exists(lab_path):
                logger.warning(f"{lab_path} does not exist")
                continue
            lab_nifti = nib.load(str(lab_path))
            lab_data = np.asanyarray(lab_nifti.dataobj)
            voxel_size = lab_nifti.header.get_zooms()[:3]
            voxel_volume = math.prod(voxel_size)

            if include_data:
                lesion_data[f"index_crop_{key}"] = index_crop
            try:
                has_iron = np.any((index_crop == lesion_index) & (lab_data == 2))
                lesion_stats[f"has_iron_{key}"] = has_iron

                rim = get_lesion_rim(index_crop, lab_data, lesion_index)
                rim_count = int(rim.sum())
                rim_sphere = rim_enclosing_sphere_radius(rim, voxel_size)
                hull = get_convex_hull(rim, voxel_sizes=voxel_size)

                if include_data:
                    lesion_data[f"rim_{key}"] = rim
                    lesion_data[f"rim_hull_{key}"] = hull

                lesion_stats[f"rim_voxels_{key}"] = rim_count
                lesion_stats[f"rim_volume_{key}"] = rim_count * voxel_volume
                if hull is not None:
                    lesion_stats[f"rim_hull_volume_{key}"] = hull.volume
                lesion_stats[f"rim_sphere_radius_{key}"] = rim_sphere
            except Exception:
                logger.warning(f"Rim analysis failed for sub{subid} lesion {lesion_index} ({key})")
                logger.debug(traceback.format_exc())

            try:
                lesion = get_center_lesion(index_crop, lab_data, lesion_index)
                lesion_count = int(lesion.sum())
                hull = get_convex_hull(lesion, voxel_sizes=voxel_size)

                if include_data:
                    lesion_data[f"lesion_{key}"] = lesion
                    lesion_data[f"lesion_hull_{key}"] = hull

                lesion_stats[f"lesion_voxels_{key}"] = lesion_count
                lesion_stats[f"lesion_volume_{key}"] = lesion_count * voxel_volume
                if hull is not None:
                    lesion_stats[f"lesion_hull_volume_{key}"] = hull.volume
            except Exception:
                logger.warning(f"Lesion analysis failed for sub{subid} lesion {lesion_index} ({key})")
                logger.debug(traceback.format_exc())

            if include_data:
                lesion_data[f"voxel_size_{key}"] = voxel_size

        subject_lesion_data.append(lesion_data)
        subject_lesion_stats.append(lesion_stats)
    return subject_lesion_stats, subject_lesion_data



def analyze_label(index_crop, lab_nifti: nib.Nifti1Image, lesion_index, include_data=True, key="infer"):
    key = "_" + key.removeprefix("_")
    lesion_stats = {}
    lesion_data = {}
    lab_data = np.asanyarray(lab_nifti.dataobj)
    voxel_size = lab_nifti.header.get_zooms()[:3]
    voxel_volume = math.prod(voxel_size)

    if include_data:
        lesion_data[f"index_crop{key}"] = index_crop
    has_iron = np.any((index_crop == lesion_index) & (lab_data == 2))
    lesion_stats[f"has_iron{key}"] = has_iron

    rim = get_lesion_rim(index_crop, lab_data, lesion_index)
    rim_count = int(rim.sum())
    rim_sphere = rim_enclosing_sphere_radius(rim, voxel_size)
    hull = get_convex_hull(rim, voxel_sizes=voxel_size)

    if include_data:
        lesion_data[f"rim{key}"] = rim
        lesion_data[f"rim_hull{key}"] = hull

    lesion_stats[f"rim_voxels{key}"] = rim_count
    lesion_stats[f"rim_volume{key}"] = rim_count * voxel_volume
    if hull is not None:
        lesion_stats[f"rim_hull_volume{key}"] = hull.volume
    lesion_stats[f"rim_sphere_radius{key}"] = rim_sphere

    lesion = get_center_lesion(index_crop, lab_data, lesion_index)
    lesion_count = int(lesion.sum())
    hull = get_convex_hull(lesion, voxel_sizes=voxel_size)

    if include_data:
        lesion_data[f"lesion{key}"] = lesion
        lesion_data[f"lesion_hull{key}"] = hull

    lesion_stats[f"lesion_voxels{key}"] = lesion_count
    lesion_stats[f"lesion_volume{key}"] = lesion_count * voxel_volume
    if hull is not None:
        lesion_stats[f"lesion_hull_volume{key}"] = hull.volume

    if include_data:
        lesion_data[f"voxel_size{key}"] = voxel_size
        
    if key == "_infer":
        features = {
            **compute_pca_features(lesion_data[f"rim{key}"]),
            **compute_radial_metrics(lesion_data[f"rim{key}"]),
        }
        lesion_stats.update(features)
    
    return lesion_stats, lesion_data



def analyze_subject_prl(subid, dataset, label_keys, lesion_indices=None, include_data=True, cases: pd.DataFrame | None = None):
    if isinstance(label_keys, str):
        label_keys = [label_keys]
    
    if cases is None:
        cases = dataset.cases
    elif not isinstance(cases, pd.DataFrame):
        print("cases argument must be None or pd.DataFrame, but defaulting to dataset.cases")
        cases = dataset.cases
    if lesion_indices is None:
        lesion_indices = cases.loc[subid, :].index
    subject = dataset.subject(subid)
    lesion_index_path = subject.dir / "lstai_lesion_index.nii.gz"
    lesion_index_vol = np.asanyarray(nib.load(str(lesion_index_path)).dataobj).astype(np.int32)
    # Parse bounding boxes
    cfg = dataset.preprocess
    bbox_suffix = f"xy{cfg.expand_xy}_z{cfg.expand_z}"
    bbox_file = subject.dir / f"lstai_bounding_boxes_{bbox_suffix}.txt"
    bounding_boxes = parse_bounding_boxes(bbox_file)

    subject_lesion_data = []
    subject_lesion_stats = []
    for lesion_index in lesion_indices:
        lesion_case = cases.loc[(subid, lesion_index), :]
        lesion_data = {"subid": subid, "lesion_index": lesion_index, "case_type": lesion_case.get('case_type')} 
        lesion_stats = {"subid": subid, "lesion_index": lesion_index, "case_type": lesion_case.get('case_type')} 
        coords = bounding_boxes[lesion_index - 1][1]       
        index_crop = crop_from_volume(lesion_index_vol, coords)
        for key in label_keys:
            lab_path = lesion_case.get(key)
            if lab_path is None or not os.path.exists(lab_path):
                logger.warning(f"{lab_path} does not exist")
                raise
            lab_nifti = nib.load(str(lab_path))
            try:
                lesion_s, lesion_d = analyze_label(index_crop, lab_nifti, lesion_index)
            except Exception:
                logger.warning(f"Rim analysis failed for sub{subid} lesion {lesion_index} ({key})")
                logger.debug(traceback.format_exc())
                lesion_s = lesion_d = None
            lesion_stats.update(lesion_s)
            lesion_data.update(lesion_d)

        subject_lesion_data.append(lesion_data)
        subject_lesion_stats.append(lesion_stats)
    return subject_lesion_stats, subject_lesion_data




# refactor so this can take a subid instead of a case so that lstai_lesion_index doesn't have to 
#   be loaded like a 100 different times
def analyze_prl_case(case, dataset, 
                     include_data=True,
                     screen_iron=False,
                     count_rim=False,
                     ):
    """Full analysis of a single PRL case.

    Computes rim and lesion statistics for both ground truth and inference,
    including voxel counts, volumes, convex hull volumes, and enclosing sphere radii.

    Args:
        case: pd.Series row from dataset.cases or experiment.cases.
            Must have: image, plus index=(subid, lesion_index).
            label and inference are both optional — at least one must exist on disk.
        dataset: Dataset instance (provides data_root, preprocess config, subject lookup).
        include_data: If True (default), return arrays (rim masks, hulls, etc.)
            in lesion_data. If False, lesion_data contains only subid/lesion_index
            — useful for batch runs to avoid holding large arrays in memory.

    Returns:
        (lesion_stats, lesion_data) tuple:
            lesion_stats: Dict of scalar metrics for this case.
            lesion_data: Dict of arrays (rim masks, hulls, etc.) for visualization,
                or just identifiers if include_data=False.
        Returns None if neither label nor inference exists on disk.
    """
    subid, lesion_index = case.name if hasattr(case, "name") else (case["subid"], case["lesion_index"])
    subject = dataset.subject(subid)

    lesion_index_path = subject.dir / "lstai_lesion_index.nii.gz"
    lesion_index_vol = nib.load(str(lesion_index_path)).get_fdata().astype(np.int32)

    # Parse bounding boxes
    cfg = dataset.preprocess
    bbox_suffix = f"xy{cfg.expand_xy}_z{cfg.expand_z}"
    bbox_file = subject.dir / f"lstai_bounding_boxes_{bbox_suffix}.txt"
    bounding_boxes = parse_bounding_boxes(bbox_file)

    try:
        assert bounding_boxes[lesion_index - 1][0] == lesion_index
    except (AssertionError, IndexError):
        logger.warning(f"Bounding box index mismatch for sub{subid} lesion {lesion_index}")
    coords = bounding_boxes[lesion_index - 1][1]

    groundtruth_path = case.get("label")
    inference_path = case.get("inference")

    label_paths = {}
    if groundtruth_path is not None:
        gt_path = Path(groundtruth_path)
        if gt_path.exists():
            label_paths["truth"] = gt_path
        else:
            logger.debug(f"Ground truth not found: {gt_path}")
    if inference_path is not None:
        inf_path = Path(inference_path)
        if inf_path.exists():
            label_paths["infer"] = inf_path
        else:
            logger.debug(f"Inference output not found: {inf_path}")

    if not label_paths:
        logger.warning(f"No label or inference found for sub{subid} lesion {lesion_index}")
        return None

    lesion_stats = {
        "subid": subid,
        "lesion_index": lesion_index,
    }
    lesion_data = {"subid": subid, "lesion_index": lesion_index}

    for key, lab_path in label_paths.items():
        lab_nifti = nib.load(str(lab_path))
        lab_data = lab_nifti.get_fdata().astype(np.uint8)
        voxel_size = lab_nifti.header.get_zooms()[:3]
        voxel_volume = math.prod(voxel_size)

        if include_data:
            lesion_data[f"label_{key}"] = lab_data

        index_crop = crop_from_volume(lesion_index_vol, coords)
        if include_data:
            lesion_data[f"index_crop_{key}"] = index_crop

        try:
            has_iron = np.any((index_crop == lesion_index) & (lab_data == 2))
            if screen_iron:
                return has_iron
            if key == "infer":
                lesion_stats["has_iron_infer"] = has_iron

            rim = get_lesion_rim(index_crop, lab_data, lesion_index)
            rim_count = int(rim.sum())
            rim_sphere = rim_enclosing_sphere_radius(rim, voxel_size)
            hull = get_convex_hull(rim, voxel_sizes=voxel_size)

            if include_data:
                lesion_data[f"rim_{key}"] = rim
                lesion_data[f"rim_hull_{key}"] = hull

            lesion_stats[f"rim_voxels_{key}"] = rim_count
            lesion_stats[f"rim_volume_{key}"] = rim_count * voxel_volume
            if hull is not None:
                lesion_stats[f"rim_hull_volume_{key}"] = hull.volume
            lesion_stats[f"rim_sphere_radius_{key}"] = rim_sphere
        except Exception:
            logger.warning(f"Rim analysis failed for sub{subid} lesion {lesion_index} ({key})")
            logger.debug(traceback.format_exc())

        try:
            lesion = get_center_lesion(index_crop, lab_data, lesion_index)
            lesion_count = int(lesion.sum())
            hull = get_convex_hull(lesion, voxel_sizes=voxel_size)

            if include_data:
                lesion_data[f"lesion_{key}"] = lesion
                lesion_data[f"lesion_hull_{key}"] = hull

            lesion_stats[f"lesion_voxels_{key}"] = lesion_count
            lesion_stats[f"lesion_volume_{key}"] = lesion_count * voxel_volume
            if hull is not None:
                lesion_stats[f"lesion_hull_volume_{key}"] = hull.volume
        except Exception:
            logger.warning(f"Lesion analysis failed for sub{subid} lesion {lesion_index} ({key})")
            logger.debug(traceback.format_exc())

        if include_data:
            lesion_data[f"voxel_size_{key}"] = voxel_size

    return lesion_stats, lesion_data
