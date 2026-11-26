import subprocess
import signal
import multiprocessing.pool
from pathlib import Path
import os

from functools import partial


def command(cmd, dry_run=False, suppress=False, verbose=False, env=None):
    if dry_run or verbose:
        print("Executing: %s" % cmd)
        if dry_run:
            return 0
    if suppress:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=env)
        _, stderr = p.communicate()
        if stderr:
            print(stderr)
        return p.returncode
    # sp_cmd = cmd.split(' ')
    # return subprocess.call(sp_cmd)
    return subprocess.run(
        cmd, shell=True, env=env, check=True, capture_output=True, text=True
    )


def run_if_missing(
    file, cmd, dry_run=False, suppress=False, verbose=False, env=None, bypass=False
):
    if not file.exists() or bypass:
        return command(
            cmd, dry_run=dry_run, suppress=suppress, verbose=verbose, env=env
        )


def convert_to_winroot(path: Path):
    return Path("H:/") / path.relative_to("/mnt/h")


def open_itksnap_workspace_cmd(
    images: list[str], labels: list[str] = None, win=False, rename_root: tuple = None
):
    """
    images: full paths to the image files
    labels: full paths to segmentations to load
    win: when I'm on WSL, I prefer to open through my windows pwsh prompt, so this 
        will convert wsl drive roots to normal windows
    rename_root: when I'm working through a VScode tunnel, I can't open GUIs, 
        so I use this to replace rename_root[0] in the paths with rename_root[1],
        then I can just copy the command locally. slightly less tedious than 
        copying everything each time
    """
    if images is None:
        raise Exception("No images")
    if labels is None:
        labels = []
    images_tmp = images.copy()
    images = [Path(im) for im in images if im.exists()]
    labels = [Path(lab) for lab in labels if lab.exists()]

    if len(images) == 0:
        print(f"Couldn't find {images_tmp}")
        return
    if win:
        images = [convert_to_winroot(Path(p)) for p in images]
        labels = [convert_to_winroot(Path(p)) for p in labels]
    elif rename_root:
        images = [Path(rename_root[1]) / p.relative_to(rename_root[0]) for p in images]
        labels = [Path(rename_root[1]) / p.relative_to(rename_root[0]) for p in labels]
    
            
    images = [str(p) for p in images]
    labels = [str(p) for p in labels]
    command = ["itksnap"]
    command.extend(["-g", images[0]])
    # command.extend(" ".join(["-o {}".format(im) for im in images[1:]]).split(" "))
    if len(images) > 1:
        command.append("-o")
        command.extend(images[1:])
    if len(labels) > 0:
        command.append("-s")
        command.extend(labels)
    return " ".join(command)


def open_fsleyes_cmd(
    images: list[str], labels: list[str] = None, rename_root: tuple = None
):
    """
    images: full paths to the image files
    labels: full paths to segmentations to load
    win: when I'm on WSL, I prefer to open through my windows pwsh prompt, so this 
        will convert wsl drive roots to normal windows
    rename_root: when I'm working through a VScode tunnel, I can't open GUIs, 
        so I use this to replace rename_root[0] in the paths with rename_root[1],
        then I can just copy the command locally. slightly less tedious than 
        copying everything each time
    """
    lut_file = Path("/media/smbshare/srs-9/prl_project/configs/fsleyes_prl.lut")
    if images is None:
        raise Exception("No images")
    if labels is None:
        labels = []
    if rename_root:
        images = [Path(rename_root[1]) / p.relative_to(rename_root[0]) for p in images]
        labels = [Path(rename_root[1]) / p.relative_to(rename_root[0]) for p in labels]
        lut_file = Path(rename_root[1]) / lut_file.relative_to(rename_root[0])
    images = [str(p) for p in images]
    labels = [str(p) for p in labels]
    

    cmd = [
        "fsleyes",
        "-xh",
        "-yh",
    ]
    for im in images:
        cmd_part = [im, "-ot", "volume"]
        if "phase" in im:
            cmd_part.extend(["-dr", "-300", "300"])
        cmd.extend(cmd_part)
    
    for lab in labels:
        cmd_part = [lab, "-ot", "label", "-l", str(lut_file)]
        cmd.extend(cmd_part)
    
    return " ".join(cmd)

def make_screenshot(img_path, seg_path, out_path, colormap="red-yellow", alpha=0.7):
    """
    Render center slice of img with seg overlay using FSLeyes headless.
    """
    cmd = [
        "fsleyes",
        "render",
        "--outfile",
        out_path,
        "--size",
        "800",
        "800",
        "--scene",
        "ortho",
        "--hideCursor",
        "--hideLabels",
        "-xh",
        "-yh",
        img_path,
        "-ot",
        "volume",
        "-dr",
        "-300",
        "300",
        seg_path,
        "--overlayType",
        "label",
        "--outline",
        # or use "volume" with a colormap if it's a probability/soft seg:
        # seg_path, "--overlayType", "volume", "--cmap", colormap, "--alpha", str(alpha * 100),
    ]
    print(" ".join([str(part) for part in cmd]))
    subprocess.run(cmd, check=True, env=os.environ.copy())


# # Batch
# pairs = [
#     ("/path/to/img1.nii.gz", "/path/to/seg1.nii.gz"),
#     ("/path/to/img2.nii.gz", "/path/to/seg2.nii.gz"),
# ]

# for img, seg in pairs:
#     subj = os.path.basename(img).replace(".nii.gz", "")
#     make_screenshot(img, seg, f"/path/to/out/{subj}_qc.png")
