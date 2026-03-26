"""Diagnostics for PRL inference outputs.

Standalone module — no coupling to inference.py or the CLI.
Can be called as a library or run directly as a script.

Usage:
    python inference_diagnostics.py /path/to/run_dir /path/to/subject_dir [--data-root PATH]
"""

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from loguru import logger


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


def _get_lesion_rim(
    index_crop: np.ndarray, infer_data: np.ndarray, lesion_id: int,
    n_dilate: int = 1,
) -> np.ndarray:
    """Get boolean mask of rim voxels belonging to a specific lesion.

    Uses connected components to find label=2 regions that overlap with the
    central lesion's footprint (dilated to capture rim just outside boundary).
    Excludes components belonging to distant neighbor lesions.

    Args:
        index_crop: Cropped lstai_lesion_index (same shape as infer_data).
        infer_data: Inference output (0=bg, 1=lesion, 2=rim).
        lesion_id: The central lesion's integer ID.
        n_dilate: Voxels to dilate the lesion footprint by.

    Returns:
        Boolean mask (same shape as infer_data) of rim voxels for this lesion.
    """
    rim_mask = infer_data == 2
    labeled, n_components = ndimage.label(rim_mask)

    lesion_mask = index_crop == lesion_id
    dilated = ndimage.binary_dilation(lesion_mask, iterations=n_dilate)

    # Keep only components that overlap with the dilated lesion mask
    result = np.zeros_like(rim_mask)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if np.any(comp_mask & dilated):
            result |= comp_mask

    return result


def _count_rim_for_lesion(
    index_crop: np.ndarray, infer_data: np.ndarray, lesion_id: int,
    n_dilate: int = 1,
) -> int:
    """Count rim voxels belonging to a specific lesion."""
    return int(_get_lesion_rim(index_crop, infer_data, lesion_id, n_dilate).sum())


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


def count_predicted_prls(
    subject_dir: Path,
    expand_xy: int,
    expand_z: int,
    infer_output_dir: Path,
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
    subject_dir = Path(subject_dir)
    infer_output_dir = Path(infer_output_dir)
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
        infer_path = infer_output_dir / subject_rel / str(index) / infer_filename
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
