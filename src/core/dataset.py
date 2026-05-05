"""Dataset class — owns data identity, fold assignments, and resolved case paths.

A Dataset is identified by name (e.g. "roi_train2"). All paths, configs,
and defaults are derived from that name. The ``cases`` property provides
a DataFrame of all cases with fully resolved paths — no Experiment needed
for path lookups.
"""

from __future__ import annotations

import os
import re
import random
import json
import pandas as pd
import numpy as np
from pathlib import Path
from functools import cached_property
from math import floor

from loguru import logger

from helpers.paths import (
    PROJECT_ROOT, DATA_ROOT, TRAIN_ROOT,
    load_config,
)
from core.configs import PreprocessingConfig, AlgoConfig


class Dataset:
    """Represents a named dataset with fixed subjects and fold assignments.

    All paths are derived from the dataset name:
      - train_home:    PROJECT_ROOT/training (parent of all datasets)
      - dataset_home:  PROJECT_ROOT/training/{name} (templates, dataset.yaml)
      - work_home:     TRAIN_ROOT/{name} (run directories with model outputs)
      - data_root:     DATA_ROOT (subject imaging data)

    Pass ``preprocess`` to override the default preprocessing config for
    path resolution. When omitted, ``default_preprocess`` from dataset.yaml
    is used.
    """

    def __init__(self, name: str, preprocess: PreprocessingConfig | None = None):
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
        self.prl_df_path = Path(config.get("prl_df", None))
        self.subjects_path = Path(config["subjects"])

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

        self._preprocess = preprocess

    # --- Preprocessing config ---

    @property
    def preprocess(self) -> PreprocessingConfig:
        """Active preprocessing config (explicit override or dataset default)."""
        return self._preprocess or self.default_preprocess

    # --- Lazy-loaded data ---

    @cached_property
    def prl_df(self) -> pd.DataFrame | None:
        if self.prl_df_path is None:
            return None
        prl_df = pd.read_csv(self.prl_df_path, index_col="subid")
        prl_df['date_mri'] = prl_df['date_mri'].astype("Int64")
        return prl_df

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

    # --- Datalist ---

    @property
    def datalist_template_path(self) -> Path:
        return self.dataset_home / "datalist_template.json"

    @cached_property
    def datalist_template(self) -> dict:
        with open(self.datalist_template_path, 'r') as f:
            datalist_template = json.load(f)
        return datalist_template

    # --- Cases DataFrame ---

    @cached_property
    def cases(self) -> pd.DataFrame:
        """DataFrame of all cases with fully resolved absolute paths.

        Columns: subid, lesion_index, split, case_type, image, label, subject_dir
        Index: (subid, lesion_index)

        Paths are constructed from the datalist template using the active
        preprocessing config (image prefix + expansion suffix). The prepared
        datalist doesn't need to exist on disk — paths are derived from the
        template's directory structure.
        """
        return self._build_cases_df()

    def _build_cases_df(self) -> pd.DataFrame:
        template = self.datalist_template
        cfg = self.preprocess
        image_prefix = ".".join(sorted(cfg.images)) + "_"
        bbox_suffix = f"xy{cfg.expand_xy}_z{cfg.expand_z}"

        rows = []

        # Testing cases
        for item in template.get("testing", []):
            case_dir = self.data_root / item["image"]
            rows.append({
                "subid": item["subid"],
                "lesion_index": item["lesion_index"],
                "split": "testing",
                "case_type": _resolve_case_type(item["label"]),
                "image": case_dir / f"{image_prefix}{bbox_suffix}.nii.gz",
                "label": self.data_root / f"{item['label']}{bbox_suffix}.nii.gz",
                "subject_dir": case_dir.parent,
            })

        # Training cases (have a fold number)
        for item in template.get("training", []):
            case_dir = self.data_root / item["image"]
            rows.append({
                "subid": item["subid"],
                "lesion_index": item["lesion_index"],
                "split": f"fold{item['fold']}",
                "case_type": _resolve_case_type(item["label"]),
                "image": case_dir / f"{image_prefix}{bbox_suffix}.nii.gz",
                "label": self.data_root / f"{item['label']}{bbox_suffix}.nii.gz",
                "subject_dir": case_dir.parent,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index(["subid", "lesion_index"])
        return df

    # --- Subject lookup ---

    def subject_session(self, subid) -> str:
        return f"sub{subid}-{self.prl_df.loc[subid, 'date_mri']}"

    def subject(self, subid) -> Subject:
        """Get a Subject handle for a given subject ID."""
        return Subject(subid, self)

    # --- Datalist creation ---

    def create_datalist(self, rebuild: bool = False) -> Path | None:
        """Create datalist_template.json with stratified fold assignments.

        Idempotent unless rebuild=True. The template is image-agnostic —
        it stores directory paths, not stacked-image prefixes. Image stack
        composition is determined later by Experiment.prepare_data().

        Returns:
            Path to the written datalist_template.json, or None if skipped.
        """
        output_path = self.datalist_template_path
        label_info_path = output_path.with_name("label_info.json")

        if output_path.exists() and not rebuild:
            logger.info(f"{output_path} exists; use rebuild=True to replace it")
            return None

        # Normalize suffix dict: ensure leading underscore
        suffix_to_use = dict(self.suffix_to_use)  # copy
        for k, suffix in suffix_to_use.items():
            if len(suffix) > 1 and suffix[0] != "_":
                suffix_to_use[k] = "_" + suffix

        # Scan subject directories and categorize PRL vs lesion-only
        prl_folders = []
        lesion_folders = []
        for subid in self.subjects:
            subid = int(subid)
            sesid = self.prl_df.loc[subid, "date_mri"]
            subject_root = self.data_root / f"sub{subid}-{sesid}"
            prl_labels = set(
                int(self.prl_df.loc[subid, f"PRL{i}_label"])
                for i in range(1, 21)
                if self.prl_df.loc[subid, f"confidence.{i-1}"] in ["definite", "probable"]
            )

            folders = [
                Path(item.path) for item in os.scandir(subject_root)
                if item.is_dir() and re.match(r"^\d+", item.name)
            ]
            for folder in folders:
                index = int(folder.name)
                if index < 1:
                    continue
                if index in prl_labels:
                    prl_folders.append((folder, subid, index))
                else:
                    lesion_folders.append((folder, subid, index))

        def _make_entry(folder, subid, index, case_type, suffix=""):
            if len(suffix) > 0 and suffix[0] != "_":
                suffix = "_" + suffix
            rel = str(folder.relative_to(self.data_root))
            if case_type == "PRL":
                label = f"{rel}/prl_label{suffix}_"
            else:
                label = f"{rel}/lesion_"
            return {
                "subid": subid, "lesion_index": index,
                "image": f"{rel}/",
                "label": label,
                "case_type": case_type,
            }

        datalist: dict[str, list] = {"training": [], "testing": []}

        for group, case_type in [(prl_folders, "PRL"), (lesion_folders, "Lesion")]:
            inds = list(range(len(group)))
            random.shuffle(inds)
            test_end = floor(len(inds) * self.test_split)

            for i in range(test_end):
                folder, subid, index = group[inds[i]]
                suffix = suffix_to_use.get(subid, "") if case_type == "PRL" else ""
                entry = _make_entry(folder, subid, index, case_type, suffix)
                datalist["testing"].append(entry)

            for i in range(test_end, len(inds)):
                folder, subid, index = group[inds[i]]
                suffix = suffix_to_use.get(subid, "") if case_type == "PRL" else ""
                entry = _make_entry(folder, subid, index, case_type, suffix)
                entry["fold"] = i % self.n_folds
                datalist["training"].append(entry)

        with open(output_path, "w") as f:
            json.dump(datalist, f, indent=4)

        label_types = {
            "prl_labels": [
                str(item[0] / f"prl_label{suffix_to_use.get(item[1], '')}_")
                for item in prl_folders
            ],
            "lesion_labels": [str(item[0]) for item in lesion_folders],
        }
        with open(label_info_path, "w") as f:
            json.dump(label_types, f, indent=4)

        return output_path

    def __repr__(self) -> str:
        pp = self.preprocess
        return (
            f"Dataset('{self.name}', "
            f"images={list(pp.images)}, "
            f"expand_xy={pp.expand_xy}, expand_z={pp.expand_z})"
        )

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


def _resolve_case_type(label_path: str) -> str:
    """Determine case type from the label filename."""
    if "prl" in Path(label_path).name:
        return "PRL"
    return "Lesion"


class Subject:
    """Lightweight handle for a single subject's paths and data.

    Provides easy access to subject-level files and filtered cases
    without loading everything upfront.
    """

    def __init__(self, subid: int, dataset: Dataset):
        self.subid = subid
        self.dataset = dataset
        self.session = dataset.subject_session(subid)
        self.dir = dataset.data_root / self.session

    @cached_property
    def bounding_boxes(self) -> list[tuple[int, list[int]]]:
        """Parse bounding boxes for this subject at the dataset's expansion params."""
        bbox_file = self.dir / f"lstai_bounding_boxes_{self.dataset.preprocess.suffix}.txt"
        bounding_boxes = []
        with open(bbox_file) as f:
            for line in f:
                parts = line.split()
                index = int(parts[0])
                coords = list(map(int, parts[1:]))
                bounding_boxes.append((index, coords))
        return bounding_boxes

    @property
    def cases(self) -> pd.DataFrame:
        """All cases for this subject from the dataset."""
        return self.dataset.cases.loc[self.subid]

    def load_nifti(self, name: str):
        """Load a top-level NIfTI from the subject directory.

        Args:
            name: Filename stem (e.g. 'flair', 'phase', 'lstai_lesion_index').
                  '.nii.gz' is appended if not present.
        """
        import nibabel as nib
        if not name.endswith(".nii.gz"):
            name = f"{name}.nii.gz"
        return nib.load(str(self.dir / name))

    def __repr__(self) -> str:
        return f"Subject({self.subid}, dir='{self.dir.name}')"