"""Stack multi-channel images and finalize datalist with expansion suffixes.

Takes the datalist_template.json (with path prefixes) and produces
datalist_xy{X}_z{Z}.json with complete paths including the expansion suffix.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from loguru import logger

from helpers.shell_interface import run_if_missing
from helpers.paths import load_config

curr_dir = Path(__file__).parent
CONCAT_SH = str(Path(__file__).parent.parent.parent / "preprocessing" / "concatImages.sh")


def prepare_training_data(datalist_template_path, data_root, images,
                           expand_xy, expand_z, output_path):
    """Stack channels and produce the final datalist with full paths.

    The template stores image-agnostic directory paths (e.g.
    "sub1038-20161031/1/"). This function computes the stacked image
    prefix from ``images`` and appends the expansion suffix to produce
    complete paths.

    Args:
        datalist_template_path: Path to datalist_template.json.
        data_root: Root data directory.
        images: List of image names (e.g. ["flair", "phase"]).
        expand_xy: X/Y expansion parameter.
        expand_z: Z expansion parameter.
        output_path: Path to write the final datalist JSON.

    Returns:
        Path to the written datalist file.
    """
    data_root = Path(data_root)
    with open(datalist_template_path, "r") as f:
        datalist = json.load(f)

    image_basenames = [im.removesuffix(".nii.gz") for im in sorted(images)]
    image_prefix = ".".join(image_basenames) + "_"
    bbox_suffix = f"xy{expand_xy}_z{expand_z}"
    image_names = [f"{im}_{bbox_suffix}.nii.gz" for im in image_basenames]
    logger.info(
        f"Ensuring [{', '.join(images)}] stack and adding \"{bbox_suffix}\" suffix "
        f"to {len(datalist['training'])} training cases and {len(datalist['testing'])} testing cases"
    )
    for subset in ["training", "testing"]:
        for a_case in tqdm(datalist[subset], total=len(datalist[subset])):
            # "image" is a directory path relative to data_root, e.g. "sub1038-20161031/1/"
            case_dir = data_root / a_case["image"]

            image_stack = case_dir / f"{image_prefix}{bbox_suffix}.nii.gz"
            input_images = [str(case_dir / im) for im in image_names]
            input_images_arg = " ".join(input_images)
            run_if_missing(
                image_stack,
                f"bash {CONCAT_SH} {image_stack} {input_images_arg}",
            )
            a_case["image"] = str(image_stack)
            a_case["label"] = str(
                data_root / (a_case["label"] + f"{bbox_suffix}.nii.gz")
            )

            if not os.path.exists(a_case["label"]):
                raise FileNotFoundError(f"Label not found: {a_case['label']}")

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(datalist, f, indent=4)

    return output_path


# --- CLI wrapper (backward compatibility) ---

def make_argument_parser(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawTextHelpFormatter,
        description="Stack multi-channel images and finalize datalist",
    )
    parser.add_argument("config", type=str, help="Path to config file")
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
    config = load_config(args.config)

    train_home = Path(config["train_home"])
    expand_xy, expand_z = config["expand_xy"], config["expand_z"]

    prepare_training_data(
        datalist_template_path=train_home / "datalist_template.json",
        data_root=Path(config["dataroot"]),
        images=config["images"],
        expand_xy=expand_xy,
        expand_z=expand_z,
        output_path=train_home / f"datalist_xy{expand_xy}_z{expand_z}.json",
    )


if __name__ == "__main__":
    main()
