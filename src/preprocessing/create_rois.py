"""
Create ROI crops around lesion bounding boxes using FSL.

Fixing the issue where if a lesion had a prl, it wouldn't even look at the
lst-ai lesion segmentation, so there were cases (e.g. sub1038) where big parts
of large lesions were missing from the training.

A potential problem here is that if the rings exceed the extent of the lesion,
the bounding boxes could exclude some of the ring if the expansion isn't enough.
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

from loguru import logger

from helpers.shell_interface import command, run_if_missing
from helpers.parallel import BetterPool
from helpers.paths import load_config
from preprocessing.verify_segmentations import verify_prl

curr_dir = Path(__file__).parent

DEFINE_BOXES_SH = str(curr_dir / "define_bounding_boxes.sh")

IMAGES = [
    "flair.nii.gz",
    "phase.nii.gz",
    "t1.nii.gz",
]
LESION_MASK = "space-flair_seg-lst.nii.gz"


def prepare_prl(lesion_path, rim_path, output_path, dry_run=False, bypass=False, exception_ok=True):
    """Combine lesion (index 1) and rim (index 2) into a single PRL label."""
    run_if_missing(
        output_path,
        f"c3d {lesion_path} -binarize {rim_path} -binarize -scale 2 -max -o {output_path}",
        dry_run=dry_run,
        bypass=False,
    )

    if not verify_prl(output_path):
        logger.warning("COULD NOT VERIFY PRL")
        logger.warning(f"c3d {lesion_path} -binarize {rim_path} -binarize -scale 2 -max -o {output_path}")
        if not exception_ok:
            raise Exception


def ensure_ring_seg(subject_root, suffix=None, dry_run=False):
    """Extract ring segmentation from full PRL probability mask."""
    full_seg_path = subject_root / f"prl_mask_def_prob_{suffix}.nii.gz"
    if suffix is None:
        for suffix in ["CH", "SRS", "LR"]:
            full_seg_path = subject_root / f"prl_mask_def_prob_{suffix}.nii.gz"
            if full_seg_path.exists():
                break
        else:
            raise FileNotFoundError(f"No segmentation in {subject_root}")

    ring_seg_path = subject_root / f"prl_rim_def_prob_{suffix}.nii.gz"
    if ring_seg_path.exists():
        return ring_seg_path

    command(f"fslmaths {full_seg_path} -thr 2 -uthr 2 {ring_seg_path}", dry_run=dry_run)
    return ring_seg_path


def prepare_rois(subid, suffix, prl_df, expand_xy, expand_z, data_root, dry_run=False):
    """Crop ROIs for a single subject.

    Args:
        data_root: Root directory containing subject folders. Passed explicitly
            instead of using a module-level global.
    """
    sesid = prl_df.loc[subid, "date_mri"]
    subject_root = data_root / f"sub{subid}-{sesid}"

    bounding_box_file = (
        subject_root / f"lstai_bounding_boxes_xy{expand_xy}_z{expand_z}.txt"
    )
    if not bounding_box_file.exists():
        command(
            f"bash {DEFINE_BOXES_SH} --expand-xy {expand_xy} --expand-z {expand_z} {subject_root}",
        )

    with open(bounding_box_file, "r") as f:
        bounding_boxes = []
        for line in f.readlines():
            parts = line.split(" ")
            bounding_boxes.append((int(parts[0]), " ".join(parts[1:]).strip()))

    # Some confluent lesions will have two rims, so two PRL will have the same lesion label and
    # will end up as one item in the training set
    prl_labels = set(
        [
            int(prl_df.loc[subid, f"PRL{i}_label"])
            for i in range(1, 21)
            if prl_df.loc[subid, f"confidence.{i-1}"] in ["definite", "probable"]
        ]
    )
    for index, box in bounding_boxes:
        output_path = subject_root / str(index)
        output_path.mkdir(exist_ok=True)

        for input_name in IMAGES:
            input_basename = input_name.removesuffix(".nii.gz")
            output_name = f"{input_basename}_xy{expand_xy}_z{expand_z}.nii.gz"
            run_if_missing(
                output_path / output_name,
                f"fslroi {subject_root / input_name} {output_path / output_name} {box}",
                dry_run=dry_run,
            )

        lesion_in = subject_root / LESION_MASK
        lesion_out = output_path / f"lesion_xy{expand_xy}_z{expand_z}.nii.gz"
        run_if_missing(lesion_out, f"fslroi {lesion_in} {lesion_out} {box}", dry_run=dry_run)

        ring_in = ensure_ring_seg(subject_root, suffix=suffix, dry_run=dry_run)
        ring_basename = ring_in.name.removesuffix(".nii.gz")
        ring_outname = f"{ring_basename}_xy{expand_xy}_z{expand_z}.nii.gz"
        if index in prl_labels:
            ring_out = output_path / ring_outname
            run_if_missing(
                ring_out,
                f"fslroi {ring_in} {ring_out} {box}",
                dry_run=dry_run,
                verbose=True,
            )
            final_prl_path = output_path / f"prl_label_{suffix}_xy{expand_xy}_z{expand_z}.nii.gz"
            try:
                prepare_prl(lesion_out, ring_out, final_prl_path, dry_run=dry_run)
            except Exception:
                logger.error("FAILED")
                logger.error(f"{lesion_out} {ring_out} {final_prl_path}")
                command(f"rm {final_prl_path}")
                raise


def _prepare_roi_wrapper(args):
    """Wrapper for multiprocessing — unpacks dict args."""
    return prepare_rois(
        subid=args["subid"],
        suffix=args["suffix"],
        prl_df=args["prl_df"],
        expand_xy=args["expand_xy"],
        expand_z=args["expand_z"],
        data_root=args["data_root"],
        dry_run=args["dry_run"],
    )


def create_rois_for_subjects(subjects, suffix_to_use, prl_df, data_root,
                              expand_xy, expand_z, processes=None, dry_run=False):
    """Create ROI crops for a list of subjects.

    This is the primary API for Dataset.create_rois(). All parameters are
    explicit — no globals.
    """
    const_args = dict(
        prl_df=prl_df, expand_xy=expand_xy, expand_z=expand_z,
        data_root=data_root, dry_run=dry_run,
    )

    if processes is None:
        for subid in tqdm(subjects):
            task = {"subid": subid, "suffix": suffix_to_use[subid], **const_args}
            _prepare_roi_wrapper(task)
    else:
        logger.info(f"Starting {len(subjects)} tasks with {processes} processes")
        tasks = [
            {"subid": subid, "suffix": suffix_to_use[subid], **const_args}
            for subid in subjects
        ]
        with BetterPool(processes) as pool:
            results_iterator = pool.imap_unordered(_prepare_roi_wrapper, tasks)
            for _ in tqdm(results_iterator, total=len(subjects)):
                pass


# --- CLI wrapper (backward compatibility) ---

def make_argument_parser(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawTextHelpFormatter,
        description="Create ROIs",
    )
    parser.add_argument("label_config", type=str, help="Path to label_config file")
    parser.add_argument(
        "--processes", type=int, default=None,
        help="Number of processes for multithreading",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="To do dry run",
    )
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = make_argument_parser(argv)
    args, _ = parser.parse_known_args()
    label_config = load_config(args.label_config)

    prl_df = pd.read_csv(label_config["prl_df"], index_col="subid")

    with open(label_config["subjects"], "r") as f:
        subjects = [int(line.strip()) for line in f.readlines()]

    suffix_to_use = {}
    with open(label_config["suffix_to_use"], "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            subid, suffix = line.strip().split(",")
            suffix_to_use[int(subid)] = suffix

    create_rois_for_subjects(
        subjects=subjects,
        suffix_to_use=suffix_to_use,
        prl_df=prl_df,
        data_root=Path(label_config["dataroot"]),
        expand_xy=label_config["expand_xy"],
        expand_z=label_config["expand_z"],
        processes=args.processes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
