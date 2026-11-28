"""Fan-in postprocessing: reassemble per-lesion ROI predictions into full-brain volumes.

Ported from src/scripts/inference.py::uncrop_predictions() as a standalone
module with no imports from src/.

Typical usage:

    from prlsegresnet.scripts.postprocess import uncrop_predictions

    output_path = uncrop_predictions(
        subject_dir=Path("/data/sub1038-20161031"),
        prediction_paths={1: Path(".../1/pred.nii.gz"), 2: Path(".../2/pred.nii.gz")},
        expand_xy=20,
        expand_z=2,
    )
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from loguru import logger

from .preprocess import load_bboxes


def uncrop_predictions(
    subject_dir: Path,
    prediction_paths: dict[int, Path],
    expand_xy: int = 20,
    expand_z: int = 2,
    reference_image: str = "flair.nii.gz",
    output_name: str | None = None,
    bboxes: list[dict] | None = None,
) -> Path:
    """Reassemble per-ROI predictions into a full-brain volume.

    Overlapping regions are merged with np.maximum.

    Args:
        subject_dir: Subject directory containing the reference image and bbox file.
        prediction_paths: {lesion_id: Path} mapping to per-ROI prediction NIfTIs.
        expand_xy: Bounding box expansion in X/Y (for finding bbox file).
        expand_z: Bounding box expansion in Z.
        reference_image: Filename of the reference image for shape/affine.
        output_name: Output filename. Defaults to "prl_inference.nii.gz".
        bboxes: Pre-loaded bounding boxes. If None, reads from text file.

    Returns:
        Path to the saved full-brain prediction NIfTI.
    """
    subject_dir = Path(subject_dir)
    if output_name is None:
        output_name = "prl_inference.nii.gz"

    # Reference image for shape and affine
    ref_nii = nib.load(str(subject_dir / reference_image))
    brain_shape = ref_nii.shape[:3]
    out_data = np.zeros(brain_shape, dtype=np.uint8)

    # Load bounding boxes
    if bboxes is None:
        bbox_suffix = f"xy{expand_xy}_z{expand_z}"
        bbox_file = subject_dir / f"lstai_bounding_boxes_{bbox_suffix}.txt"
        bboxes = load_bboxes(bbox_file)

    bbox_by_id = {bb["lesion_id"]: bb for bb in bboxes}

    for lid, pred_path in prediction_paths.items():
        pred_path = Path(pred_path)
        if not pred_path.exists():
            logger.warning(f"Prediction not found: {pred_path}")
            continue

        if lid not in bbox_by_id:
            logger.warning(f"No bounding box for lesion {lid}")
            continue

        bbox = bbox_by_id[lid]
        crop_data = nib.load(str(pred_path)).get_fdata().astype(np.uint8)

        # Place crop back into full-brain volume, handling out-of-bounds.
        roi_start = bbox["roi_start"]
        roi_size = bbox["roi_size"]

        for dim, (start, size, brain_dim) in enumerate(
            zip(roi_start, roi_size, brain_shape)
        ):
            crop_start = max(0, -start)
            brain_start = max(0, start)
            usable = min(size - crop_start, brain_dim - brain_start)

            if dim == 0:
                cx0, bx0, nx = crop_start, brain_start, usable
            elif dim == 1:
                cy0, by0, ny = crop_start, brain_start, usable
            else:
                cz0, bz0, nz = crop_start, brain_start, usable

        roi_slice = crop_data[cx0:cx0 + nx, cy0:cy0 + ny, cz0:cz0 + nz]
        np.maximum(
            out_data[bx0:bx0 + nx, by0:by0 + ny, bz0:bz0 + nz],
            roi_slice,
            out=out_data[bx0:bx0 + nx, by0:by0 + ny, bz0:bz0 + nz],
        )

    output_path = subject_dir / output_name
    out_img = nib.Nifti1Image(out_data, ref_nii.affine, ref_nii.header)
    nib.save(out_img, str(output_path))
    logger.info(f"Saved uncropped prediction: {output_path}")
    return output_path


def uncrop_from_arrays(
    predictions: dict[int, np.ndarray],
    bboxes: list[dict],
    brain_shape: tuple[int, ...],
) -> np.ndarray:
    """In-memory version: reassemble per-ROI arrays into a full-brain array.

    Args:
        predictions: {lesion_id: prediction_array} mapping.
        bboxes: Bounding box dicts (from compute_lesion_bboxes or load_bboxes).
        brain_shape: (X, Y, Z) shape of the full-brain volume.

    Returns:
        Full-brain numpy array (uint8).
    """
    out_data = np.zeros(brain_shape, dtype=np.uint8)
    bbox_by_id = {bb["lesion_id"]: bb for bb in bboxes}

    for lid, crop_data in predictions.items():
        if lid not in bbox_by_id:
            continue

        bbox = bbox_by_id[lid]
        roi_start = bbox["roi_start"]
        roi_size = bbox["roi_size"]

        for dim, (start, size, brain_dim) in enumerate(
            zip(roi_start, roi_size, brain_shape)
        ):
            crop_start = max(0, -start)
            brain_start = max(0, start)
            usable = min(size - crop_start, brain_dim - brain_start)

            if dim == 0:
                cx0, bx0, nx = crop_start, brain_start, usable
            elif dim == 1:
                cy0, by0, ny = crop_start, brain_start, usable
            else:
                cz0, bz0, nz = crop_start, brain_start, usable

        roi_slice = crop_data[cx0:cx0 + nx, cy0:cy0 + ny, cz0:cz0 + nz]
        np.maximum(
            out_data[bx0:bx0 + nx, by0:by0 + ny, bz0:bz0 + nz],
            roi_slice,
            out=out_data[bx0:bx0 + nx, by0:by0 + ny, bz0:bz0 + nz],
        )

    return out_data
