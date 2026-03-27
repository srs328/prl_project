"""Diagnostics for PRL inference outputs.

Standalone module — no coupling to inference.py or the CLI.
Can be called as a library or run directly as a script.

Usage:
    python inference_diagnostics.py /path/to/run_dir /path/to/subject_dir [--data-root PATH]
"""
import math
import matplotlib.pyplot as plt
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from loguru import logger
import traceback


def _parse_bounding_boxes(bbox_file: Path) -> list[tuple[int, list[int]]]:
    """Parse bounding box file into list of (index, [xmin, xsize, ymin, ysize, zmin, zsize])."""
    bounding_boxes = []
    with open(bbox_file) as f:
        for line in f:
            parts = line.split()
            index = int(parts[0])
            coords = list(map(int, parts[1:]))
            bounding_boxes.append((index, coords))
    return bounding_boxes


def _crop_from_volume(volume: np.ndarray, coords: list[int]) -> np.ndarray:
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

    # Allocate output (zero = background/no lesion)
    crop = np.zeros((xsize, ysize, zsize), dtype=volume.dtype)

    # Compute valid ranges for each dimension
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
            # Entirely outside the brain — return zeros
            return crop

        slices_brain.append(slice(b_start, b_end))
        slices_crop.append(slice(c_start, c_end))

    crop[tuple(slices_crop)] = volume[tuple(slices_brain)]
    return crop


def _get_center_lesion(
        index_crop: np.ndarray, label_data: np.ndarray, lesion_id: int,
) -> np.ndarray:
    """Get an array where just the voxels of the center lesion are true

    Args:
        index_crop (np.ndarray): cropped roi containing
        lesion_id (int): index of center lesion based on the lesion index map 

    Returns:
        np.ndarray: mask of center lesion (ndarray of true/false)
    """
    lesion_mask = label_data == 1
    labeled, n_components = ndimage.label(lesion_mask)

    index_mask = index_crop == lesion_id

    # Keep only components that overlap with the indexed mask
    result = np.zeros_like(lesion_mask)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if np.any(comp_mask & index_mask):
            result |= comp_mask

    return result


def get_center_label(
    index_crop: np.ndarray, label_data: np.ndarray, lesion_id: int,
    label_class: int, n_dilate: int | None = None,
) -> np.ndarray:
    """Get boolean mask of segmented voxels associated with a specific lesion.

    Uses connected components to find label=label_class regions that overlap with the
    central lesion's footprint (optionally dilated to capture rim just outside boundary).
    Excludes components belonging to distant neighbor lesions.

    Args:
        index_crop: Cropped lstai_lesion_index (same shape as label_data).
        label_data: Inference output (0=bg, 1=lesion, 2=rim).
        lesion_id: The central lesion's integer ID.
        label_class: Label value to extract (1=lesion, 2=rim)
        n_dilate (optional): Voxels to dilate the lesion footprint by.

    Returns:
        Boolean mask (same shape as label_data) of rim voxels for this lesion.
    """
    
    label_mask = label_data == label_class
    labeled, n_components = ndimage.label(label_mask)
    
    index_mask = index_crop == lesion_id
    if n_dilate is not None:
        index_mask = ndimage.binary_dilation(index_mask, iterations=n_dilate)
    
    # Keep only components that overlap with the indexed mask
    result = np.zeros_like(label_mask)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if np.any(comp_mask & index_mask):
            result |= comp_mask

    return result


def _get_lesion_rim(
    index_crop: np.ndarray, label_data: np.ndarray, lesion_id: int,
    n_dilate: int = 1,
) -> np.ndarray:
    """Get boolean mask of rim voxels belonging to a specific lesion.

    Uses connected components to find label=2 regions that overlap with the
    central lesion's footprint (dilated to capture rim just outside boundary).
    Excludes components belonging to distant neighbor lesions.

    Args:
        index_crop: Cropped lstai_lesion_index (same shape as label_data).
        label_data: Inference output (0=bg, 1=lesion, 2=rim).
        lesion_id: The central lesion's integer ID.
        n_dilate: Voxels to dilate the lesion footprint by.

    Returns:
        Boolean mask (same shape as label_data) of rim voxels for this lesion.
    """
    rim_mask = label_data == 2
    labeled, n_components = ndimage.label(rim_mask)

    index_mask = index_crop == lesion_id
    dilated = ndimage.binary_dilation(index_mask, iterations=n_dilate)

    # Keep only components that overlap with the dilated lesion mask
    result = np.zeros_like(rim_mask)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if np.any(comp_mask & dilated):
            result |= comp_mask

    return result


"""See if this would do the same
    # 1. Find all labeled IDs that overlap with your index_mask
# (We cast index_mask to bool to use it as a coordinate selector)
overlapping_ids = np.unique(labeled[index_mask.astype(bool)])

# 2. Remove 0 (the background label) if it exists in the set
overlapping_ids = overlapping_ids[overlapping_ids != 0]

# 3. Create the final mask by checking which pixels match these IDs
result = np.isin(labeled, overlapping_ids)
"""


def _count_rim_for_lesion(
    index_crop: np.ndarray, label_data: np.ndarray, lesion_id: int,
    n_dilate: int = 1,
) -> int:
    """Count rim voxels belonging to a specific lesion."""
    return int(_get_lesion_rim(index_crop, label_data, lesion_id, n_dilate).sum())


def get_convex_hull(mask: np.ndarray, voxel_sizes: tuple[float, ...] = None) -> np.ndarray:
    """Convex hull of binary mask voxels in mm³.

    Returns None if fewer than 4 non-coplanar rim voxels (ConvexHull needs this).
    """
    coords = np.argwhere(mask)  # (N, 3) in voxel indices
    if len(coords) < 4:
        return None
    
    if voxel_sizes is not None:
        coords = coords * np.array(voxel_sizes)
    try:
        hull = ConvexHull(coords)
    except Exception:
        # Degenerate geometry (coplanar points, etc.)
        return None
    
    return hull


def rim_convex_hull_volume(rim_mask: np.ndarray, voxel_sizes: tuple[float, ...]) -> float | None:
    """Convex hull volume of rim voxels in mm³.

    Returns None if fewer than 4 non-coplanar rim voxels (ConvexHull needs this).
    """
    coords = np.argwhere(rim_mask)  # (N, 3) in voxel indices
    if len(coords) < 4:
        return None
    coords_mm = coords * np.array(voxel_sizes)
    try:
        hull = ConvexHull(coords_mm)
        return float(hull.volume)
    except Exception:
        # Degenerate geometry (coplanar points, etc.)
        return None


def rim_enclosing_sphere_radius(rim_mask: np.ndarray, voxel_sizes: tuple[float, ...]) -> float | None:
    """Radius (mm) of the smallest sphere enclosing all rim voxels.

    Uses convex hull vertices + centroid approach (guaranteed enclosing,
    not mathematically minimal but close for typical rim shapes).
    Returns None if no rim voxels.
    """
    coords = np.argwhere(rim_mask)
    if len(coords) == 0:
        return None
    coords_mm = coords * np.array(voxel_sizes)
    if len(coords) < 4:
        # Too few points for ConvexHull — use all points directly
        center = coords_mm.mean(axis=0)
        return float(np.max(np.linalg.norm(coords_mm - center, axis=1)))
    try:
        hull = ConvexHull(coords_mm)
        vertices = coords_mm[hull.vertices]
    except Exception:
        vertices = coords_mm
    center = vertices.mean(axis=0)
    return float(np.max(np.linalg.norm(vertices - center, axis=1)))



def rim_enclosing_sphere_radius0(rim_mask: np.ndarray, voxel_sizes: tuple[float, ...]) -> float | None:
    """Radius (mm) of the smallest sphere enclosing all rim voxels.

    Uses convex hull vertices + centroid approach (guaranteed enclosing,
    not mathematically minimal but close for typical rim shapes).
    Returns None if no rim voxels.
    """
    coords = np.argwhere(rim_mask)
    if len(coords) == 0:
        return None
    coords_mm = coords * np.array(voxel_sizes)
    if len(coords) < 4:
        # Too few points for ConvexHull — use all points directly
        center = coords_mm.mean(axis=0)
        return float(np.max(np.linalg.norm(coords_mm - center, axis=1)))
    try:
        hull = ConvexHull(coords_mm)
        vertices = coords_mm[hull.vertices]
    except Exception:
        vertices = coords_mm
    center = vertices.mean(axis=0)
    return float(np.max(np.linalg.norm(vertices - center, axis=1)))


def count_predicted_prls(
    case_metadata: dict,
    expand_xy: int,
    expand_z: int,
    data_root: Path,
    images: tuple[str, ...],
) -> dict:
    """Count how many lesions were predicted as PRL for a subject.

    Uses lstai_lesion_index.nii.gz to definitively assign rim voxels
    to their source lesion, avoiding false counts from overlapping
    bounding boxes.

    Returns:
        Dict with subject name, total lesions, predicted PRL count,
        PRL lesion indices, and rim voxel counts per PRL.
    """
    subject_dir = Path(case_metadata['image'].parent.parent)
    data_root = Path(data_root)

    # Load lesion index once
    lesion_index_path = subject_dir / "lstai_lesion_index.nii.gz"
    lesion_index = nib.load(str(lesion_index_path)).get_fdata().astype(np.int32)

    # Parse bounding boxes
    bbox_suffix = f"xy{expand_xy}_z{expand_z}"
    bbox_file = subject_dir / f"lstai_bounding_boxes_{bbox_suffix}.txt"
    bounding_boxes = _parse_bounding_boxes(bbox_file)

    # TODO generalize this for ground truth: add postfix parameter that can be used to build any filename
    # Build inference output filename pattern
    image_basenames = sorted(images)
    image_prefix = ".".join(image_basenames) + "_"
    infer_filename = f"{image_prefix}{bbox_suffix}_infer.nii.gz"
    subject_rel = subject_dir.relative_to(data_root)

    prls = []

    for index, coords in bounding_boxes:
        # Load inference output for this ROI
        infer_path = None
        if not infer_path.exists():
            logger.warning(f"Inference output not found: {infer_path}")
            continue

        infer_data = nib.load(str(infer_path)).get_fdata().astype(np.uint8)

        # Crop lesion index with the same bounding box
        index_crop = _crop_from_volume(lesion_index, coords)

        # PRL detection: rim voxels that overlap the central lesion's footprint
        is_prl = np.any((index_crop == index) & (infer_data == 2))

        if is_prl:
            rim = _get_lesion_rim(index_crop, infer_data, index)
            # Get voxel sizes from the inference nifti header
            vox = nib.load(str(infer_path)).header.get_zooms()[:3]
            prls.append({
                "lesion_index": index,
                "rim_voxels": int(rim.sum()),
                "hull_volume_mm3": rim_convex_hull_volume(rim, vox),
                "sphere_radius_mm": rim_enclosing_sphere_radius(rim, vox),
            })

    return {
        "subject": subject_dir.name,
        "total_lesions": len(bounding_boxes),
        "predicted_prls": len(prls),
        "prls": prls,
    }
    
    
def analyze_prl_case(prl_case: dict, experiment, bbox_suffix):
    subid = prl_case['subid']
    index = prl_case['lesion_index']
    subject_dir = experiment.dataset.subject_dir(subid)
    data_root = experiment.dataset.data_root

    lesion_index_path = subject_dir / "lstai_lesion_index.nii.gz"
    lesion_index = nib.load(str(lesion_index_path)).get_fdata().astype(np.int32)

    # Parse bounding boxes
    expand_xy = experiment.preprocess_config.expand_xy
    expand_z = experiment.preprocess_config.expand_z
    bbox_suffix = f"xy{expand_xy}_z{expand_z}"
    bbox_file = subject_dir / f"lstai_bounding_boxes_{bbox_suffix}.txt"
    bounding_boxes = _parse_bounding_boxes(bbox_file)

    try:
        assert bounding_boxes[index-1][0] == index
    except AssertionError:
        print(index, bounding_boxes[index-1][0])
    coords = bounding_boxes[index-1][1]

    # Load inference output for this ROI
    groundtruth_path = data_root / prl_case["label"]
    infer_path = data_root / prl_case["inference"]
    if not infer_path.exists():
        print(f"Inference output not found: {infer_path}")
        return None
    
    label_keys = ["truth", "infer"]
    lesion_stats = {"lesion_index": index, **{f"rim_voxels_{k}": None for k in label_keys}}
    lesion_data = {"subid": subid, "lesion_index": index}
    for id, lab in zip(label_keys, [groundtruth_path, infer_path]):
        lab_nifti = nib.load(str(lab))
        lab_data = lab_nifti.get_fdata().astype(np.uint8)
        voxel_size = lab_nifti.header.get_zooms()[:3]
        voxel_volume = math.prod(voxel_size)

        lesion_data[f"label_{id}"] = lab_data

        # Crop lesion index with the same bounding box
        index_crop = _crop_from_volume(lesion_index, coords)
        lesion_data[f"index_crop_{id}"] = index_crop

        try:
            # any iron detection: rim voxels that overlap the central lesion's footprint
            has_iron = np.any((index_crop == index) & (lab_data == 2))
            if id == "infer":
                lesion_stats['has_iron_infer'] = has_iron
            # ---Process Rim---
            # get the rim for the center lesion
            rim = _get_lesion_rim(index_crop, lab_data, index)
            rim_count = int(rim.sum())
            rim_sphere_radius = rim_enclosing_sphere_radius(rim, voxel_size)

            # get convex hull
            hull = get_convex_hull(rim, voxel_sizes=voxel_size)
            
            lesion_data.update({
                f"rim_{id}": rim,
                f"rim_hull_{id}": hull,
            })

            lesion_stats.update({
                f"rim_voxels_{id}": rim_count,
                f"rim_volume_{id}": rim_count*voxel_volume,
            })

            lesion_stats.update({
                f"rim_hull_volume_{id}": hull.volume,
                f"rim_sphere_radius_{id}": rim_sphere_radius,
            })
        except Exception:
            print(prl_case['subid'], prl_case['lesion_index'])
            tb_str = traceback.format_exc()
            print(f"Captured Traceback:\n{tb_str}")
            pass

        # ---Process T2 Lesion---
        try:
            lesion = _get_center_lesion(index_crop, lab_data, index)
            lesion_count = int(lesion.sum())

            # get convex hull
            hull = get_convex_hull(lesion, voxel_sizes=voxel_size)
            
            lesion_data.update({
                f"lesion_{id}": lesion,
                f"lesion_hull_{id}": hull,
            })

            lesion_stats.update({
                f"lesion_voxels_{id}": lesion_count,
                f"lesion_volume_{id}": lesion_count*voxel_volume,
            })
            lesion_stats.update({
                f"lesion_hull_volume_{id}": hull.volume
            })
        except Exception:
            tb_str = traceback.format_exc()
            print(f"Captured Traceback:\n{tb_str}")
            pass

        lesion_data[f"voxel_size_{id}"] = voxel_size    
    return lesion_stats, lesion_data


def plot_lesion_rim_3d(
    prl_case: dict, label_source: str = "infer",
    alpha_wireframe: float = 0.15, alpha_lesion: float = 0.05,
    alpha_hull: float = 0.4, alpha_rim: float = 0.3
):
    klab = label_source
    rim = prl_case[f"rim_{klab}"]
    voxel_size = prl_case[f'voxel_size_{klab}']
    # Rim voxel coordinates in mm
    coords = np.argwhere(rim) * np.array(voxel_size)

    # Convex hull
    hull = prl_case[f"rim_hull_{klab}"]

    # Enclosing sphere (same approach as the function)
    vertices = coords[hull.vertices]
    center = vertices.mean(axis=0)
    radius = np.max(np.linalg.norm(vertices - center, axis=1))

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Rim voxels
    ax.scatter(*coords.T, c='red', alpha=alpha_rim, s=10, label='Rim voxels')

    # Convex hull faces
    for simplex in hull.simplices:
        pts = coords[simplex]
        # Close the triangle
        tri = np.vstack([pts, pts[0]])
        ax.plot(*tri.T, 'b-', alpha=alpha_hull, linewidth=0.5)

    # Enclosing sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='green', alpha=alpha_wireframe, linewidth=0.3)

    # Also show lesion voxels for context (label=1 from infer_data)
    lesion_coords = np.argwhere(prl_case[f'lesion_{klab}'] == 1) * np.array(voxel_size)
    # lesion_coords = np.argwhere(lab_data == 1) * np.array(voxel_size)
    ax.scatter(*lesion_coords.T, c='grey', alpha=alpha_lesion, s=2, label='Lesion')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()
    ax.set_title(f'Hull vol={hull.volume:.1f} mm³, sphere r={radius:.2f} mm')
    plt.tight_layout()
    plt.show()



def run_diagnostics(
    subject_dir: Path,
    expand_xy: int,
    expand_z: int,
    infer_output_dir: Path,
    data_root: Path,
    images: tuple[str, ...],
) -> dict:
    """Run all inference diagnostics for a subject.

    Currently includes PRL count. Extensible for future diagnostics.
    """
    results = count_predicted_prls(
        subject_dir, expand_xy, expand_z,
        infer_output_dir, data_root, images,
    )
    return results


def print_diagnostics(results: dict) -> None:
    """Pretty-print diagnostic results."""
    print(f"\n{'='*60}")
    print(f"Inference Diagnostics: {results['subject']}")
    print(f"{'='*60}")
    print(f"Total lesions:    {results['total_lesions']}")
    print(f"Predicted PRLs:   {results['predicted_prls']}")

    if results["prls"]:
        print("\nPRL lesions:")
        for prl in results["prls"]:
            hull = prl.get("hull_volume_mm3")
            sphere = prl.get("sphere_radius_mm")
            hull_str = f"{hull:.1f} mm³" if hull is not None else "N/A"
            sphere_str = f"{sphere:.2f} mm" if sphere is not None else "N/A"
            print(f"  Lesion {prl['lesion_index']:>3d}: "
                  f"{prl['rim_voxels']} rim voxels | "
                  f"hull={hull_str} | sphere r={sphere_str}")
    else:
        print("\nNo PRLs predicted.")
    print()


# --- Script entry point ---

if __name__ == "__main__":
    import argparse

    from helpers.paths import load_config
    from core.dataset import Dataset

    parser = argparse.ArgumentParser(
        description="Run inference diagnostics on a subject"
    )
    parser.add_argument("run_dir", type=Path, help="Trained model run directory")
    parser.add_argument("subject_dir", type=Path, help="Subject directory")
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Data root override (default: dataset.yaml data_root)",
    )
    args = parser.parse_args()

    label_config = load_config(args.run_dir / "label_config.json")
    expand_xy = label_config["expand_xy"]
    expand_z = label_config["expand_z"]
    images = tuple(label_config.get("images", ["flair", "phase"]))

    ds = Dataset(label_config["dataset_name"])
    data_root = args.data_root or ds.data_root
    subject_dir = args.subject_dir
    if not subject_dir.is_absolute():
        subject_dir = data_root / subject_dir

    infer_output_dir = subject_dir / "inference_output"

    results = run_diagnostics(
        subject_dir, expand_xy, expand_z,
        infer_output_dir, data_root, images,
    )
    print_diagnostics(results)
