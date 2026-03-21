"""Dataset class — owns data identity and preprocessing pipeline.

A Dataset is identified by name (e.g. "roi_train2"). All paths, configs,
and defaults are derived from that name. The preprocessing pipeline
(create ROIs, create datalist, prepare training data) is exposed as methods.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from functools import cached_property

from helpers.paths import (
    PROJECT_ROOT, DATA_ROOT, TRAIN_ROOT,
    load_dataset_config,
)
from core.configs import PreprocessingConfig, TrainingConfig
from preprocessing.create_rois import create_rois_for_subjects
from preprocessing.create_datalist import create_datalist_template
from preprocessing.prepare_training_data import prepare_training_data


class Dataset:
    """Represents a named dataset with fixed subjects and fold assignments.

    All paths are derived from the dataset name:
      - source_home: PROJECT_ROOT/training/{name} (templates, dataset.yaml)
      - work_home:   TRAIN_ROOT/{name} (run directories with model outputs)
      - data_root:   DATA_ROOT (subject imaging data)
    """

    def __init__(self, name: str):
        self.name = name
        self.source_home = PROJECT_ROOT / "training" / name
        self.work_home = TRAIN_ROOT / name
        self.data_root = DATA_ROOT

        config = load_dataset_config(name)
        self._config = config

        self.images = config["images"]
        self.n_folds = config["n_folds"]
        self.test_split = config["test_split"]
        self.prl_df_path = Path(config["prl_df"])
        self.subjects_path = Path(config["subjects"])
        self.suffix_to_use_path = Path(config["suffix_to_use"])

        # Parse defaults
        defaults = config.get("defaults", {})
        self.default_preprocess = PreprocessingConfig(
            expand_xy=defaults.get("expand_xy", 20),
            expand_z=defaults.get("expand_z", 2),
        )
        training_defaults = defaults.get("training", {})
        self.default_training = TrainingConfig.from_dict(training_defaults)

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
        with open(self.suffix_to_use_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                subid, suffix = line.strip().split(",")
                result[int(subid)] = suffix
        return result

    @property
    def datalist_template_path(self) -> Path:
        return self.source_home / "datalist_template.json"

    def datalist_path(self, config: PreprocessingConfig | None = None) -> Path:
        """Path to the final datalist for given expansion parameters."""
        if config is None:
            config = self.default_preprocess
        return self.source_home / f"datalist_{config.suffix}.json"

    # --- Preprocessing pipeline ---

    def create_rois(self, config: PreprocessingConfig | None = None) -> None:
        """Crop ROIs for all subjects at given expand_xy/expand_z."""
        if config is None:
            config = self.default_preprocess
        create_rois_for_subjects(
            subjects=self.subjects,
            suffix_to_use=self.suffix_to_use,
            prl_df=self.prl_df,
            data_root=self.data_root,
            expand_xy=config.expand_xy,
            expand_z=config.expand_z,
            processes=config.processes,
            dry_run=config.dry_run,
        )

    def create_datalist(self, rebuild: bool = False) -> Path | None:
        """Create datalist_template.json with fold assignments.

        Idempotent unless rebuild=True. Returns path to the template, or
        None if it already exists and rebuild is False.
        """
        return create_datalist_template(
            subjects=self.subjects,
            suffix_to_use=self.suffix_to_use,
            prl_df=self.prl_df,
            data_root=self.data_root,
            images=self.images,
            n_folds=self.n_folds,
            test_split=self.test_split,
            output_path=self.datalist_template_path,
            rebuild=rebuild,
        )

    def prepare_data(self, config: PreprocessingConfig | None = None) -> Path:
        """Stack channels and produce datalist_xy{X}_z{Z}.json."""
        if config is None:
            config = self.default_preprocess
        return prepare_training_data(
            datalist_template_path=self.datalist_template_path,
            data_root=self.data_root,
            images=self.images,
            expand_xy=config.expand_xy,
            expand_z=config.expand_z,
            output_path=self.datalist_path(config),
        )

    def preprocess(self, config: PreprocessingConfig | None = None,
                   rebuild_datalist: bool = False) -> Path:
        """Full pipeline: create_rois -> create_datalist -> prepare_data.

        Returns path to the final datalist file.
        """
        if config is None:
            config = self.default_preprocess
        self.create_rois(config)
        self.create_datalist(rebuild=rebuild_datalist)
        return self.prepare_data(config)

    def __repr__(self) -> str:
        return f"Dataset('{self.name}')"
