"""Run trained PRL models on fresh subjects.

Fresh subjects have FLAIR/phase MRI and LST-AI lesion segmentations but
no PRL annotations — no prl_mask_def_prob_* files, no rater suffixes.

Usage:
    prl infer /path/to/run_dir sub9999-20250101 --data-root /path/to/data
    prl infer /path/to/run_dir --all --data-root /path/to/inference_data
    prl infer /path/to/run_dir --subjects-file subjects.txt --data-root /path/to/data
"""

import json
import os
import re
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from loguru import logger

from helpers.paths import load_config, DATA_ROOT
from helpers.shell_interface import command, run_if_missing

# Reuse the same shell scripts as the training pipeline
_PREPROCESSING_DIR = Path(__file__).parent.parent / "preprocessing"
DEFINE_BOXES_SH = str(_PREPROCESSING_DIR / "define_bounding_boxes.sh")
CONCAT_SH = str(_PREPROCESSING_DIR / "concatImages.sh")

LESION_MASK = "space-flair_seg-lst.nii.gz"


def create_rois_for_inference(
    subject_dir: Path,
    images: tuple[str, ...],
    expand_xy: int,
    expand_z: int,
    dry_run: bool = False,
) -> list[tuple[int, str]]:
    """Crop ROIs for a fresh subject (no annotations needed).

    Generates bounding boxes from lstai_lesion_index.nii.gz, then crops
    each image channel and the lesion mask for every lesion ROI.

    Returns:
        List of (index, bounding_box_string) tuples that were processed.
    """
    subject_dir = Path(subject_dir)
    bbox_suffix = f"xy{expand_xy}_z{expand_z}"
    bbox_file = subject_dir / f"lstai_bounding_boxes_{bbox_suffix}.txt"

    # Generate bounding boxes if needed
    if not bbox_file.exists():
        command(
            f"bash {DEFINE_BOXES_SH} --expand-xy {expand_xy} --expand-z {expand_z} {subject_dir}",
            verbose=True
        )

    # Parse bounding boxes
    bounding_boxes = []
    with open(bbox_file) as f:
        for line in f:
            parts = line.split()
            index = int(parts[0])
            box = " ".join(parts[1:])
            bounding_boxes.append((index, box))

    image_names = [f"{im}.nii.gz" for im in sorted(images)]

    for index, box in bounding_boxes:
        output_path = subject_dir / str(index)
        output_path.mkdir(exist_ok=True)

        # Crop each image channel
        for img_name in image_names:
            img_basename = img_name.removesuffix(".nii.gz")
            out_name = f"{img_basename}_{bbox_suffix}.nii.gz"
            run_if_missing(
                output_path / out_name,
                f"fslroi {subject_dir / img_name} {output_path / out_name} {box}",
                dry_run=dry_run,
            )

        # Crop lesion mask
        lesion_in = subject_dir / LESION_MASK
        lesion_out = output_path / f"lesion_{bbox_suffix}.nii.gz"
        run_if_missing(
            lesion_out,
            f"fslroi {lesion_in} {lesion_out} {box}",
            dry_run=dry_run,
        )

    logger.info(f"Created {len(bounding_boxes)} ROIs for {subject_dir.name}")
    return bounding_boxes


def prepare_inference_data(
    subject_dir: Path,
    images: tuple[str, ...],
    expand_xy: int,
    expand_z: int,
    dry_run: bool = False,
) -> None:
    """Stack image channels for each ROI subdirectory.

    Creates multi-channel NIfTI stacks (e.g. flair.phase_xy20_z2.nii.gz)
    matching the format used during training.
    """
    subject_dir = Path(subject_dir)
    image_basenames = sorted(images)
    image_prefix = ".".join(image_basenames) + "_"
    bbox_suffix = f"xy{expand_xy}_z{expand_z}"

    roi_dirs = sorted(
        p for p in subject_dir.iterdir()
        if p.is_dir() and re.match(r"^\d+$", p.name)
    )

    for roi_dir in roi_dirs:
        stack_name = f"{image_prefix}{bbox_suffix}.nii.gz"
        image_stack = roi_dir / stack_name
        input_images = [
            str(roi_dir / f"{im}_{bbox_suffix}.nii.gz")
            for im in image_basenames
        ]
        input_images_arg = " ".join(input_images)
        run_if_missing(
            image_stack,
            f"bash {CONCAT_SH} {image_stack} {input_images_arg}",
            dry_run=dry_run,
        )

    logger.info(f"Prepared image stacks for {len(roi_dirs)} ROIs")


def create_inference_datalist(
    subject_dir: Path,
    data_root: Path,
    images: tuple[str, ...],
    expand_xy: int,
    expand_z: int,
) -> Path:
    """Create inference datalist for a single subject.

    Scans subject_dir for numeric subdirectories (lesion ROIs) and writes
    a datalist with a "testing" key for MONAI ensemble inference.
    Paths are relative to data_root.

    Returns:
        Path to the written datalist (subject_dir / inference_datalist.json).
    """
    subject_dir = Path(subject_dir)
    data_root = Path(data_root)

    image_basenames = sorted(images)
    image_prefix = ".".join(image_basenames) + "_"
    bbox_suffix = f"xy{expand_xy}_z{expand_z}"
    stack_name = f"{image_prefix}{bbox_suffix}.nii.gz"

    subject_rel = subject_dir.relative_to(data_root)

    roi_dirs = sorted(
        p for p in subject_dir.iterdir()
        if p.is_dir() and re.match(r"^\d+$", p.name)
    )

    testing_entries = []
    for roi_dir in roi_dirs:
        image_path = str(subject_rel / roi_dir.name / stack_name)
        testing_entries.append({"image": image_path})

    datalist = {"testing": testing_entries}
    output_path = subject_dir / "inference_datalist.json"
    with open(output_path, "w") as f:
        json.dump(datalist, f, indent=4)

    logger.info(
        f"Created inference datalist with {len(testing_entries)} cases: {output_path}"
    )
    return output_path


def run_ensemble_inference(
    run_dir: Path,
    datalist_path: Path,
    data_root: Path,
    output_dir: Path,
    run_id: str,
    n_fold: int = 5,
) -> None:
    """Run MONAI ensemble inference using a trained model.

    Uses AlgoEnsembleBestByFold to ensemble predictions from all folds.

    Args:
        run_dir: Trained model directory (contains segresnet_0..4).
        datalist_path: Inference datalist JSON with "testing" key.
        data_root: Root data directory for resolving relative image paths.
        output_dir: Where to write prediction NIfTIs.
        run_id: run identifier to postpend to filenames
        n_fold: Number of cross-validation folds (default 5).
    """
    from monai.apps.auto3dseg import (
        AlgoEnsembleBestByFold,
        AlgoEnsembleBuilder,
        import_bundle_algo_history,
    )

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create task config for MONAI
    task_cfg = {
        "name": "prl_inference",
        "task": "segmentation",
        "modality": "MRI",
        "datalist": str(datalist_path),
        "dataroot": str(data_root),
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=str(run_dir)
    ) as f:
        task_file = Path(f.name)
        json.dump(task_cfg, f, indent=2)

    try:
        history = import_bundle_algo_history(str(run_dir), only_trained=True)
        builder = AlgoEnsembleBuilder(history, str(task_file))
        builder.set_ensemble_method(AlgoEnsembleBestByFold(n_fold=n_fold))
        ensemble = builder.get_ensemble()

        save_params = {
            "_target_": "SaveImage",
            "output_dir": str(output_dir),
            "data_root_dir": str(data_root),
            "output_postfix": f"infer_{run_id}",
            "separate_folder": False,
        }

        logger.info(
            f"Running ensemble inference: {len(history)} algos, "
            f"output_dir={output_dir}"
        )
        ensemble(pred_param={"image_save_func": save_params})
        logger.info("Ensemble inference complete")

    finally:
        if task_file.exists():
            task_file.unlink()


def uncrop_predictions(
    subject_dir: Path,
    expand_xy: int,
    expand_z: int,
    data_root: Path,
    images: tuple[str, ...],
    run_id: str,
) -> Path:
    """Combine inferred ROI labels back onto full brain volume.

    Reads bounding boxes and places each ROI prediction into the correct
    position in a full-brain-sized array. Overlapping regions are merged
    with np.maximum.

    Returns:
        Path to the output file.
    """
    subject_dir = Path(subject_dir)
    data_root = Path(data_root)
    output_name = f"prl_inference_{run_id}"

    # Reference image for shape and affine
    ref_img = nib.load(str(subject_dir / "flair.nii.gz"))
    brain_shape = ref_img.shape[:3]
    out_data = np.zeros(brain_shape, dtype=np.uint8)

    # Parse bounding boxes
    bbox_suffix = f"xy{expand_xy}_z{expand_z}"
    bbox_file = subject_dir / f"lstai_bounding_boxes_{bbox_suffix}.txt"

    bounding_boxes = []
    with open(bbox_file) as f:
        for line in f:
            parts = line.split()
            index = int(parts[0])
            coords = list(map(int, parts[1:]))
            bounding_boxes.append((index, coords))

    # Build expected inference output filename
    image_basenames = sorted(images)
    image_prefix = ".".join(image_basenames) + "_"
    subject_rel = subject_dir.relative_to(data_root)

    for index, coords in bounding_boxes:
        xmin, xsize, ymin, ysize, zmin, zsize = coords

        # Find the inference output
        infer_filename = f"{image_prefix}{bbox_suffix}_infer_{run_id}.nii.gz"
        infer_path = data_root / subject_rel / str(index) / infer_filename

        if not infer_path.exists():
            logger.warning(f"Inference output not found: {infer_path}")
            continue

        crop_data = nib.load(str(infer_path)).get_fdata().astype(np.uint8)

        # Handle clipping for negative start coords and brain boundary overflow.
        # fslroi zero-pads when bbox extends outside the volume.
        for dim, (start, size, brain_dim) in enumerate(
            zip(
                [xmin, ymin, zmin],
                [xsize, ysize, zsize],
                brain_shape,
            )
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

        roi_slice = crop_data[cx0:cx0+nx, cy0:cy0+ny, cz0:cz0+nz]
        np.maximum(
            out_data[bx0:bx0+nx, by0:by0+ny, bz0:bz0+nz],
            roi_slice,
            out=out_data[bx0:bx0+nx, by0:by0+ny, bz0:bz0+nz],
        )

    output_path = subject_dir / output_name
    out_img = nib.Nifti1Image(out_data, ref_img.affine, ref_img.header)
    nib.save(out_img, str(output_path))
    logger.info(f"Saved uncropped prediction: {output_path}")
    return output_path


def _derive_run_id(run_dir: Path, dataset_work_home: Path) -> str:
    """Derive a filename-safe run identifier from the run_dir path.

    E.g. run_dir = .../roi_train2/stage1_crop_lr_sweep/run1
         work_home = .../roi_train2
         → "stage1_crop_lr_sweep_run1"
    """
    try:
        rel = run_dir.relative_to(dataset_work_home)
        return str(rel).replace(os.sep, "_")
    except ValueError:
        # Fallback: just use the run dir name
        return run_dir.name


def infer_subject(
    run_dir: Path,
    subject_dir: Path,
    data_root: Path | None = None,
) -> Path:
    """Full inference pipeline for a single fresh subject.

    1. Load run configs (expand_xy, expand_z, images)
    2. Create ROIs (bounding boxes + crop images)
    3. Stack image channels
    4. Create inference datalist
    5. Run ensemble inference
    6. Uncrop predictions to full brain

    Returns:
        Path to the final full-brain inference output.
    """
    from core.configs import PreprocessingConfig

    run_dir = Path(run_dir)
    subject_dir = Path(subject_dir)

    # Load preprocessing params from the trained run
    label_config = load_config(run_dir / "label_config.json")
    expand_xy = label_config["expand_xy"]
    expand_z = label_config["expand_z"]
    images = tuple(label_config.get("images", ["flair", "phase"]))

    # Resolve data_root
    if data_root is None:
        data_root = DATA_ROOT
    data_root = Path(data_root)

    # Derive run identifier for output filename
    train_home = Path(label_config["train_home"])
    # work_home is TRAIN_ROOT / dataset_name
    dataset_name = train_home.name
    from helpers.paths import TRAIN_ROOT
    dataset_work_home = TRAIN_ROOT / dataset_name
    run_id = _derive_run_id(run_dir, dataset_work_home)

    logger.info(
        f"Inferring on {subject_dir.name} using run '{run_id}' "
        f"(images={images}, expand_xy={expand_xy}, expand_z={expand_z})"
    )

    # 1. Create ROIs
    print(subject_dir)
    create_rois_for_inference(
        subject_dir, images, expand_xy, expand_z,
    )

    # 2. Stack image channels
    prepare_inference_data(
        subject_dir, images, expand_xy, expand_z,
    )

    # 3. Create inference datalist
    datalist_path = create_inference_datalist(
        subject_dir, data_root, images, expand_xy, expand_z,
    )

    # 4. Run ensemble inference
    #   inference saved alongside rest of data with suffixes identifying 
    #   them as inference; MONAI takes output dir as the original dataroot
    #   and so it will save into data_root/subject_dir/lesion_index/ automatically
    infer_output_dir = data_root 
    run_ensemble_inference(
        run_dir, datalist_path, data_root, infer_output_dir, run_id
    )

    # 5. Uncrop predictions
    output_path = uncrop_predictions(
        subject_dir, expand_xy, expand_z,
        data_root, images, run_id,
    )

    return output_path
