"""Dataset class — owns data identity and fold assignments.

A Dataset is identified by name (e.g. "roi_train2"). All paths, configs,
and defaults are derived from that name. The only pipeline method it owns
is create_datalist() (fold assignment), since folds are a property of
the dataset itself. ROI creation and data preparation are per-experiment
and live in Experiment.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from functools import cached_property
import json

from helpers.paths import (
    PROJECT_ROOT, DATA_ROOT, TRAIN_ROOT,
    load_config,
)
from core.configs import PreprocessingConfig, AlgoConfig
from preprocessing.create_datalist import create_datalist_template


class Dataset:
    """Represents a named dataset with fixed subjects and fold assignments.

    All paths are derived from the dataset name:
      - train_home:    PROJECT_ROOT/training (parent of all datasets)
      - dataset_home:  PROJECT_ROOT/training/{name} (templates, dataset.yaml)
      - work_home:     TRAIN_ROOT/{name} (run directories with model outputs)
      - data_root:     DATA_ROOT (subject imaging data)
    """
    
    # FIXME: potentially cleaner logic if the config.gets are all done in the Dataset.load_config
    def __init__(self, name: str):
        self.name = name
        config = Dataset.load_config(name)
        self.train_home = config.get("train_home", PROJECT_ROOT / "training")
        self.dataset_home = config.get("dataset_home", 
                                       PROJECT_ROOT / "training" / name)
        self.work_home = config.get("work_home", TRAIN_ROOT / name)
        self.data_root = Path(config.get("data_root", DATA_ROOT))

        self._config = config

        self.n_folds = config["n_folds"]
        self.test_split = config["test_split"]
        self.prl_df_path = Path(config["prl_df"])
        self.subjects_path = Path(config["subjects"])
        
        # FIXME right now everything downstream might fail if suffix_to_use is None
        #   either implement the priority function I had for if its None, or just require it
        if config["suffix_to_use"] is not None:
            self.suffix_to_use_path = Path(config["suffix_to_use"])
        else:
            self.suffix_to_use_path = None
        # Parse defaults
        defaults = config.get("defaults", {})
        self.default_preprocess = PreprocessingConfig(
            images=defaults.get("images", ["flair", "phase"]),
            expand_xy=defaults.get("expand_xy", 20),
            expand_z=defaults.get("expand_z", 2),
        )
        training_defaults = defaults.get("training", {})
        if training_defaults is None:
            training_defaults = {}
        self.default_training = AlgoConfig.from_dict(training_defaults)

    @cached_property
    def prl_df(self) -> pd.DataFrame:
        return pd.read_csv(self.prl_df_path, index_col="subid")

    @cached_property
    def subjects(self) -> list[int]:
        with open(self.subjects_path, "r") as f:
            return [int(line.strip()) for line in f.readlines()]

    @cached_property
    def suffix_to_use(self) -> dict[int, str]:
        result = {}
        if self.suffix_to_use_path is None:
            return result
        with open(self.suffix_to_use_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                subid, suffix = line.strip().split(",")
                result[int(subid)] = suffix
        return result

    @property
    def datalist_template_path(self) -> Path:
        return self.dataset_home / "datalist_template.json"
    
    @cached_property
    def datalist_template(self) -> dict:
        with open(self.datalist_template_path, 'r') as f:
            datalist_template = json.load(f)
        return datalist_template
    
    def subject_session(self, subid) -> str:
        return f"sub{subid}-{self.prl_df.loc[subid, 'date_mri']}"
        
    def subject_dir(self, subid) -> Path:
        return self.data_root / self.subject_session(subid)
    
    def lesion_dir(self, datalist_case):
        return self.subject_dir(datalist_case['subid']) / str(datalist_case['lesion_index'])
    
    def get_images(self, datalist_case, images, suffix="") -> list[str]:
        im_names = [f"{im.removesuffix('.nii.gz')}{suffix}.nii.gz" for im in images]
        return [self.lesion_dir(datalist_case) / im for im in im_names]


    # I wonder if it makes sense for the main logic to be in another file and function.
    # the only reason its like this now is that I wrote create_datalist_template before
    # doing an OOP refactor
    def create_datalist(self, rebuild: bool = False) -> Path | None:
        """Create datalist_template.json with stratified fold assignments.

        Idempotent unless rebuild=True. The template is image-agnostic —
        it stores directory paths, not stacked-image prefixes. Image stack
        composition is determined later by Experiment.prepare_data().
        """
        return create_datalist_template(
            subjects=self.subjects,
            suffix_to_use=self.suffix_to_use,
            prl_df=self.prl_df,
            data_root=self.data_root,
            n_folds=self.n_folds,
            test_split=self.test_split,
            output_path=self.datalist_template_path,
            rebuild=rebuild,
        )
    
    def __repr__(self) -> str:
        return f"Dataset('{self.name}')"
    
    @staticmethod
    def load_config(name):
        """Load dataset.yaml by dataset name.

        Looks up PROJECT_ROOT/training/{name}/dataset.yaml, expands tokens,
        and resolves relative paths against the dataset's source_home directory.
        """
        dataset_home = PROJECT_ROOT / "training" / name
        config_path = dataset_home / "dataset.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' not found: {config_path} does not exist"
            )
        config = load_config(config_path)

        # Resolve relative paths against dataset_home
        for key in ("subjects", "suffix_to_use"):
            if key in config and config[key] is not None and not Path(config[key]).is_absolute():
                config[key] = str(dataset_home / config[key])
            else:
                config[key] = None

        return config

    @staticmethod
    def parse_stacked_image_name(image_name):
        """Parse image stacks that are named like im1.im2.im2_<suffix>.nii.gz
        Won't work if individual images have underscores
        """
        import re
        pattern = re.compile(r"(([A-Za-z0-10]+\.?)+?)_(.+)\.nii\.gz")
        matches = pattern.match(image_name)
        images = matches[1].split(".")
        suffix = matches[3]
        return images, suffix
        


class Subject:
    """Represents a subject and contains the various paths associated with them
    
    Should have functions to search the paths for patterns (e.g. to find an inference label)
    """