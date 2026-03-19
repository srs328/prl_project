import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import json

sys.path.append("/home/srs-9/Projects/prl_project")
from helpers.shell_interface import run_if_missing


curr_dir = Path(__file__).parent
screenshots = curr_dir / "screenshots"
CONCAT_SH = "/home/srs-9/Projects/prl_project/preprocessing/concatImages.sh"


def confidence_label(i):
    if i == 1:
        return "confidence"
    else:
        return f"confidence.{i - 1}"
    

def make_argument_parser(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawTextHelpFormatter,
        description="Create ROIs",
    )
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes for multithreading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="To do dry run",
    )
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = make_argument_parser(argv)
    args, _ = parser.parse_known_args()
    with open(args.config, "r") as f:
        config = json.load(f)


    train_home = Path(config['train_home'])
    with open(train_home/"datalist_template.json", 'r') as f:
        datalist = json.load(f)
        
    suffix_to_use = {}
    with open(config["suffix_to_use"], "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            subid, suffix = line.strip().split(",")
            suffix_to_use[int(subid)] = suffix

    expand_xy, expand_z = config["expand_xy"], config["expand_z"]
    
    image_basenames = [im.removesuffix(".nii.gz") for im in sorted(config['images'])]
    image_names = [
        f"{im}_xy{expand_xy}_z{expand_z}.nii.gz"
        for im in image_basenames
    ]

    global dataroot
    dataroot = Path(config["dataroot"])

    for subset in ["training", "testing"]:
        for a_case in tqdm(datalist[subset], total=len(datalist[subset])):
            subid = a_case['subid']
            image_stack_prefix = Path(a_case['image'])
            folder = image_stack_prefix.parent

            image_stack = folder / (image_stack_prefix.name + f"xy{expand_xy}_z{expand_z}.nii.gz")
            input_images = [str(folder / im) for im in image_names]
            input_images_arg = " ".join(input_images)
            run_if_missing(
                image_stack,
                f"bash {CONCAT_SH} {image_stack} {input_images_arg}"
            )
            a_case['image'] = str(image_stack)
            a_case['label'] = a_case['label'] + f"xy{expand_xy}_z{expand_z}.nii.gz"
            
            if not os.path.exists(a_case['label']):
                raise
            
    with open(train_home / f"datalist_xy{expand_xy}_z{expand_z}.json", 'w') as f:
        json.dump(datalist, f, indent=4)    


if __name__ == "__main__":
    main()