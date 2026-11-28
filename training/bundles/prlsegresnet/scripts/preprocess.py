"""Fan-out preprocessing: full-brain images to per-lesion ROI crops.

Replaces the FSL-based pipeline (fslstats, fslroi, fslmerge) with pure Python
using MONAI's generate_spatial_bounding_box() for bbox computation and
SpatialCrop for deterministic ROI extraction.

Typical usage from a notebook:

    from prlsegresnet.scripts.preprocess import preprocess_subject

    rois = preprocess_subject(
        subject_dir=Path("/data/sub1038-20161031"),
        images=("flair", "phase"),
        expand_xy=20,
        expand_z=2,
    )

Or batch with datalist generation:

    datalist_path = preprocess_subjects(
        subject_dirs=[...],
        output_dir=Path("/training/roi_train2/run1"),
        images=("flair", "phase"),
        expand_xy=20,
        expand_z=2,
    )
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from loguru import logger
from monai.transforms.utils import generate_spatial_bounding_box


# ---------------------------------------------------------------------------
# Bounding box computation
# ---------------------------------------------------------------------------

def compute_lesion_bboxes(
    lesion_index: np.ndarray,
    expand_xy: int = 20,
    expand_z: int = 2,
) -> list[dict]:
    """Compute one bounding box per lesion from the lesion index mask.

    Uses MONAI's generate_spatial_bounding_box with per-axis margins.
    With allow_smaller=False (default), margins can extend beyond image
    boundaries, producing negative roi_start or roi_end > image dim.
    This matches FSL's fslroi behavior (zero-pads outside the volume).

    Args:
        lesion_index: 3D integer array where each lesion has a unique nonzero label.
        expand_xy: Voxels to expand in X and Y axes.
        expand_z: Voxels to expand in Z axis.

    Returns:
        List of dicts, sorted by lesion_id, each containing:
            lesion_id: int
            roi_start: list[int] (can be negative)
            roi_end: list[int] (can exceed image dims)
            roi_size: list[int] (always positive)
            fslroi_coords: list[int] [xmin, xsize, ymin, ysize, zmin, zsize]
    """
    labels = np.unique(lesion_index)
    labels = labels[labels > 0]

    margin = [expand_xy, expand_xy, expand_z]
    bboxes = []

    for lid in sorted(labels):
        mask = (lesion_index == lid).astype(np.float32)[np.newaxis, ...]  # channel-first
        box_start, box_end = generate_spatial_bounding_box(mask, margin=margin)

        roi_start = [int(s) for s in box_start]
        roi_end = [int(e) for e in box_end]
        roi_size = [e - s for s, e in zip(roi_start, roi_end)]

        # fslroi-style coords: xmin xsize ymin ysize zmin zsize
        fslroi_coords = []
        for s, sz in zip(roi_start, roi_size):
            fslroi_coords.extend([s, sz])

        bboxes.append({
            "lesion_id": int(lid),
            "roi_start": roi_start,
            "roi_end": roi_end,
            "roi_size": roi_size,
            "fslroi_coords": fslroi_coords,
        })

    return bboxes


# ---------------------------------------------------------------------------
# Bounding box I/O (backward compat with existing text format)
# ---------------------------------------------------------------------------

def save_bboxes(bboxes: list[dict], output_path: Path) -> None:
    """Write bounding boxes in the existing text format.

    Format: one line per lesion: ``lesion_id xmin xsize ymin ysize zmin zsize``
    """
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        for bb in bboxes:
            coords = bb["fslroi_coords"]
            f.write(f"{bb['lesion_id']} {' '.join(str(c) for c in coords)}\n")


def load_bboxes(bbox_path: Path) -> list[dict]:
    """Read bounding boxes from existing text file into dict format.

    Parses the ``lesion_id xmin xsize ymin ysize zmin zsize`` format.
    """
    bbox_path = Path(bbox_path)
    bboxes = []
    with open(bbox_path) as f:
        for line in f:
            parts = line.split()
            lid = int(parts[0])
            coords = list(map(int, parts[1:]))
            xmin, xsize, ymin, ysize, zmin, zsize = coords
            roi_start = [xmin, ymin, zmin]
            roi_end = [xmin + xsize, ymin + ysize, zmin + zsize]
            roi_size = [xsize, ysize, zsize]
            bboxes.append({
                "lesion_id": lid,
                "roi_start": roi_start,
                "roi_end": roi_end,
                "roi_size": roi_size,
                "fslroi_coords": coords,
            })
    return bboxes


# ---------------------------------------------------------------------------
# Per-lesion ROI cropping
# ---------------------------------------------------------------------------

def crop_volume(volume: np.ndarray, bbox: dict) -> np.ndarray:
    """Crop a 3D volume using a bounding box, with zero-padding for out-of-bounds.

    Handles negative roi_start and roi_end beyond image dims by zero-padding,
    matching fslroi's behavior.

    Args:
        volume: 3D array (X, Y, Z) or 4D channel-first (C, X, Y, Z).
        bbox: Dict with roi_start, roi_end, roi_size keys.

    Returns:
        Cropped array of shape (*roi_size) or (C, *roi_size).
    """
    has_channels = volume.ndim == 4
    if has_channels:
        spatial = volume.shape[1:]
    else:
        spatial = volume.shape[:3]

    roi_start = bbox["roi_start"]
    roi_size = bbox["roi_size"]

    if has_channels:
        crop = np.zeros((volume.shape[0], *roi_size), dtype=volume.dtype)
    else:
        crop = np.zeros(roi_size, dtype=volume.dtype)

    slices_vol = []
    slices_crop = []
    for start, size, dim_size in zip(roi_start, roi_size, spatial):
        v_start = max(0, start)
        v_end = min(dim_size, start + size)
        c_start = max(0, -start)
        c_end = c_start + (v_end - v_start)

        if v_end <= v_start:
            return crop

        slices_vol.append(slice(v_start, v_end))
        slices_crop.append(slice(c_start, c_end))

    if has_channels:
        crop[(slice(None), *slices_crop)] = volume[(slice(None), *slices_vol)]
    else:
        crop[tuple(slices_crop)] = volume[tuple(slices_vol)]

    return crop


def crop_lesion_rois(
    subject_data: dict[str, np.ndarray],
    bboxes: list[dict],
    image_keys: list[str],
    label_key: str | None = "label",
) -> list[dict]:
    """Crop per-lesion ROIs from full-brain arrays.

    Args:
        subject_data: Dict mapping keys to 3D numpy arrays.
            Must include all image_keys. label_key is optional.
        bboxes: Output of compute_lesion_bboxes().
        image_keys: Keys to crop and stack into multi-channel image.
        label_key: Key for the label array. None to skip labels.

    Returns:
        List of dicts (one per lesion), each containing:
            image: stacked channel array (C, H, W, D)
            label: label array (H, W, D) if label_key was provided
            lesion_id: int
            bbox: the bbox dict
    """
    results = []

    for bbox in bboxes:
        roi = {"lesion_id": bbox["lesion_id"], "bbox": bbox}

        # Crop and stack image channels
        channels = []
        for key in image_keys:
            cropped = crop_volume(subject_data[key], bbox)
            channels.append(cropped)
        roi["image"] = np.stack(channels, axis=0)  # (C, H, W, D)

        # Crop label if available
        if label_key is not None and label_key in subject_data:
            roi["label"] = crop_volume(subject_data[label_key], bbox)

        results.append(roi)

    return results


def stack_channels(channel_arrays: list[np.ndarray]) -> np.ndarray:
    """Stack multiple single-channel 3D arrays into a multi-channel array.

    Replaces fslmerge -a.
    """
    return np.stack(channel_arrays, axis=0)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def save_roi_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
) -> None:
    """Save a numpy array as a NIfTI file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(output_path))


def load_subject_niftis(
    subject_dir: Path,
    image_names: tuple[str, ...] = ("flair", "phase"),
    lesion_index_name: str = "lstai_lesion_index.nii.gz",
    label_paths: dict[int, Path] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load full-brain NIfTIs for a subject.

    Args:
        subject_dir: Path to subject directory.
        image_names: Channel names (without extension). Loaded as {name}.nii.gz.
        lesion_index_name: Filename of the lesion index volume.
        label_paths: Optional mapping of lesion_id -> label NIfTI path.
            If None, no labels are loaded (inference mode).

    Returns:
        (subject_data, affine) where subject_data maps keys to 3D arrays
        and affine is from the first image.
    """
    subject_dir = Path(subject_dir)
    data = {}
    affine = None

    for name in image_names:
        nii = nib.load(str(subject_dir / f"{name}.nii.gz"))
        data[name] = nii.get_fdata().astype(np.float32)
        if affine is None:
            affine = nii.affine

    # Lesion index
    idx_nii = nib.load(str(subject_dir / lesion_index_name))
    data["lesion_index"] = idx_nii.get_fdata().astype(np.int32)

    # Labels are per-lesion, loaded later in preprocess_subject
    # (since they may be in per-lesion subdirectories)

    return data, affine


# ---------------------------------------------------------------------------
# Subject-level orchestrators
# ---------------------------------------------------------------------------

def preprocess_subject(
    subject_dir: Path,
    images: tuple[str, ...] = ("flair", "phase"),
    expand_xy: int = 20,
    expand_z: int = 2,
    label_paths: dict[int, Path] | None = None,
    save_intermediates: bool = False,
    output_dir: Path | None = None,
) -> list[dict]:
    """Full preprocessing for one subject: load images, compute bboxes, crop ROIs.

    Args:
        subject_dir: Path to subject directory containing full-brain NIfTIs.
        images: Channel names to load and stack.
        expand_xy: Bounding box expansion in X/Y.
        expand_z: Bounding box expansion in Z.
        label_paths: Optional {lesion_id: Path} mapping to per-lesion label files.
            If None, no labels are cropped (inference mode).
        save_intermediates: If True, write per-ROI NIfTI files to disk.
        output_dir: Where to save intermediates. Defaults to subject_dir.

    Returns:
        List of per-lesion dicts with image (C,H,W,D), label (H,W,D), bbox, etc.
    """
    subject_dir = Path(subject_dir)
    if output_dir is None:
        output_dir = subject_dir

    # Load full-brain volumes
    subject_data, affine = load_subject_niftis(subject_dir, image_names=images)
    lesion_index = subject_data["lesion_index"]

    # Compute bounding boxes
    bboxes = compute_lesion_bboxes(lesion_index, expand_xy, expand_z)
    logger.info(f"Found {len(bboxes)} lesions in {subject_dir.name}")

    # Load per-lesion labels if provided
    if label_paths is not None:
        for bbox in bboxes:
            lid = bbox["lesion_id"]
            if lid in label_paths:
                label_nii = nib.load(str(label_paths[lid]))
                # Labels are already cropped in existing pipeline; for full-brain
                # labels we'd crop here. For now, store per-lesion labels keyed by id.
                subject_data[f"label_{lid}"] = label_nii.get_fdata().astype(np.uint8)

    # Crop ROIs (without labels first — labels handled per-lesion below)
    rois = crop_lesion_rois(
        subject_data, bboxes,
        image_keys=list(images),
        label_key=None,
    )

    # Attach per-lesion labels if available
    if label_paths is not None:
        for roi in rois:
            lid = roi["lesion_id"]
            label_key = f"label_{lid}"
            if label_key in subject_data:
                # Label is already cropped to ROI size — use directly
                roi["label"] = subject_data[label_key]

    # Save intermediates
    bbox_suffix = f"xy{expand_xy}_z{expand_z}"
    if save_intermediates:
        save_bboxes(bboxes, output_dir / f"lstai_bounding_boxes_{bbox_suffix}.txt")

        image_prefix = ".".join(sorted(images))
        for roi in rois:
            lid = roi["lesion_id"]
            roi_dir = output_dir / str(lid)

            # Save individual channels
            for i, name in enumerate(sorted(images)):
                save_roi_nifti(
                    roi["image"][i],
                    affine,
                    roi_dir / f"{name}_{bbox_suffix}.nii.gz",
                )

            # Save stacked image
            save_roi_nifti(
                roi["image"],
                affine,
                roi_dir / f"{image_prefix}_{bbox_suffix}.nii.gz",
            )

            # Save label if available
            if "label" in roi:
                save_roi_nifti(
                    roi["label"],
                    affine,
                    roi_dir / f"label_{bbox_suffix}.nii.gz",
                )

    return rois


def preprocess_subjects(
    subject_dirs: list[Path],
    output_dir: Path,
    images: tuple[str, ...] = ("flair", "phase"),
    expand_xy: int = 20,
    expand_z: int = 2,
    label_paths_by_subject: dict[str, dict[int, Path]] | None = None,
    fold_assignments: dict[str, int | str] | None = None,
    save_intermediates: bool = True,
) -> Path:
    """Batch preprocess and generate a segresnet-compatible datalist.

    Args:
        subject_dirs: List of subject directory paths.
        output_dir: Where to write the datalist and optionally intermediates.
        images: Channel names.
        expand_xy: Bounding box expansion in X/Y.
        expand_z: Bounding box expansion in Z.
        label_paths_by_subject: Optional {subject_name: {lesion_id: label_path}}.
        fold_assignments: Optional {subject_name: fold_number_or_"testing"}.
        save_intermediates: Write per-ROI NIfTIs to disk.

    Returns:
        Path to the generated datalist JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_suffix = f"xy{expand_xy}_z{expand_z}"
    image_prefix = ".".join(sorted(images))

    training_entries = []
    testing_entries = []

    for subject_dir in subject_dirs:
        subject_dir = Path(subject_dir)
        subject_name = subject_dir.name

        subj_label_paths = None
        if label_paths_by_subject and subject_name in label_paths_by_subject:
            subj_label_paths = label_paths_by_subject[subject_name]

        subj_output = output_dir / subject_name if save_intermediates else None

        rois = preprocess_subject(
            subject_dir,
            images=images,
            expand_xy=expand_xy,
            expand_z=expand_z,
            label_paths=subj_label_paths,
            save_intermediates=save_intermediates,
            output_dir=subj_output,
        )

        for roi in rois:
            lid = roi["lesion_id"]
            roi_rel = f"{subject_name}/{lid}"

            entry = {
                "image": str(output_dir / roi_rel / f"{image_prefix}_{bbox_suffix}.nii.gz"),
            }
            if "label" in roi:
                entry["label"] = str(output_dir / roi_rel / f"label_{bbox_suffix}.nii.gz")

            # Determine fold/split
            assignment = "testing"
            if fold_assignments and subject_name in fold_assignments:
                assignment = fold_assignments[subject_name]

            if assignment == "testing":
                testing_entries.append(entry)
            else:
                entry["fold"] = int(assignment)
                training_entries.append(entry)

    datalist = {"training": training_entries, "testing": testing_entries}
    datalist_path = output_dir / f"datalist_{image_prefix}_{bbox_suffix}.json"
    with open(datalist_path, "w") as f:
        json.dump(datalist, f, indent=2)

    logger.info(
        f"Generated datalist: {len(training_entries)} training, "
        f"{len(testing_entries)} testing -> {datalist_path}"
    )
    return datalist_path
