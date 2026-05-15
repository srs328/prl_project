# %%
import logging
import os
import sys

os.environ["SEGRESNET2D_ALWAYS"] = "1"

from monai.apps.auto3dseg import (
    AlgoEnsembleBestN,
    AlgoEnsembleBuilder,
    import_bundle_algo_history,
)
from monai.utils.enums import AlgoKeys
from monai.apps.auto3dseg import AutoRunner
from monai.config import print_config

print_config()
logging.getLogger("monai.apps.auto3dseg").setLevel(logging.DEBUG)

# %%

work_dir = "/media/smbshare/srs-9/prl_project/training/roi_train2/run3"
history = import_bundle_algo_history(work_dir, only_trained=True)

# %%