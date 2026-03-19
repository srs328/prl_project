import subprocess
import signal
import multiprocessing.pool
from pathlib import Path
import os

from functools import partial

def command(cmd, dry_run=False, suppress=False, verbose=False, env=None):
    if (dry_run or verbose):
        print('Executing: %s' % cmd)
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
    return subprocess.run(cmd, shell=True, env=env, check=True, capture_output=True, text=True)


def run_if_missing(file, cmd, dry_run=False, suppress=False, verbose=False, env=None, bypass=False):
    if not file.exists() or bypass:
        return command(cmd, dry_run=False, suppress=False, verbose=False, env=None)
    


def convert_to_winroot(path: Path):
    return Path("H:/") / path.relative_to("/mnt/h")


def open_itksnap_workspace_cmd(images: list[str], labels: list[str] = None, win=False, rename_root: tuple = None):
    if images is None:
        raise Exception("No images")
    if labels is None:
        labels = []
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



def make_screenshot(img_path, seg_path, out_path, colormap="red-yellow", alpha=0.7):
    """
    Render center slice of img with seg overlay using FSLeyes headless.
    """
    cmd = [
        "fsleyes", "render",
        "--outfile", out_path,
        "--size", "800", "800",
        "--scene", "ortho",
        "--hideCursor",
        "--hideLabels",
        "-xh", "-yh",
        img_path, "-ot", "volume", "-dr", "-300", "300",
        seg_path, "--overlayType", "label", "--outline",
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


