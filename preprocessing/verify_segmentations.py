# %%
import sys
import os
import pandas as pd
from pathlib import Path

sys.path.append("/home/srs-9/Projects/prl_project")
from helpers.shell_interface import command

def fslstats_verify(seg):
    pass


def verify_prl(path):
    if not path.exists():
        return False
    result = command(
        f"fslstats {path} -R"
    ).stdout
    max_lab = int(float(result.split(" ")[1].strip()))
    if max_lab == 0:
        print(f"Segmentation {path} is empty")
    elif max_lab < 2:
        print(f"Segmentation {path} has no label above 1")
    elif max_lab > 2:
        print(f"Segmentation {path} has extraneous labels")
    else:
        return True
    return False

def verify_lesion(path):
    result = command(
        f"fslstats {path} -R"
    ).stdout
    max_lab = int(float(result.split(" ")[1].strip()))
    if max_lab == 1:
        return True
    else:
        print(f"Max label is {max_lab}")
        return False
    # "fslroi [INPUT_IMAGE] $SUBJECTDIR/prlmontage/$sessionDate/prl${lesion_label}.phase.box.nii.gz ${roi_boundaries}"

