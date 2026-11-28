"""Dataset class — owns data identity, fold assignments, and resolved case paths.

A Dataset is identified by name (e.g. "roi_train2"). All paths, configs,
and defaults are derived from that name. The ``cases`` property provides
a DataFrame of all cases with fully resolved paths — no Experiment needed
for path lookups.
"""

from __future__ import annotations

from collections import defaultdict
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

from ..helpers.paths import (
    PROJECT_ROOT, DATA_ROOT, TRAIN_ROOT,
    load_config,
)
from .configs import PreprocessingConfig, AlgoConfig

#TODO: this isnt set up for subjects with multiple sessions since Dataset assumes one sesid per subid

random.seed(42)

DEFAULT_EXPAND_XY = 20
DEFAULT_EXPAND_Z = 2
DEFAULT_IMAGES = ["flair", "phase"]

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
        preprocess_defaults = defaults.get("preprocessing", {})
        self.default_preprocess = PreprocessingConfig(
            images=preprocess_defaults.get("images", DEFAULT_IMAGES),
            expand_xy=preprocess_defaults.get("expand_xy", DEFAULT_EXPAND_XY),
            expand_z=preprocess_defaults.get("expand_z", DEFAULT_EXPAND_Z),
            prl_label_prefix=preprocess_defaults.get("prl_label_prefix", "prl_label")
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
        subject_folder_list = []
        total_prl = 0
        for subid in self.subjects:
            prl_folders = []
            lesion_folders = []
            subid = int(subid)
            sesid = self.prl_df.loc[subid, "date_mri"]
            subject_root = self.data_root / f"sub{subid}-{sesid}"
            prl_labels = {
                int(self.prl_df.loc[subid, f"PRL{i}_label"])
                for i in range(1, 21)
                if self.prl_df.loc[subid, f"confidence.{i-1}"] in ["definite", "probable"]
            }

            folders = [
                Path(item.path) for item in os.scandir(subject_root)
                if item.is_dir() and re.match(r"^\d+", item.name)
            ]
            for folder in folders:
                index = int(folder.name)
                if index < 1:
                    continue
                if index in prl_labels:
                    # prl_folders.append((folder, subid, index))
                    prl_folders.append((folder, index))
                else:
                    # lesion_folders.append((folder, subid, index))
                    lesion_folders.append((folder, index))

            subject_folder_list.append((
                subid, 
                prl_folders, 
                lesion_folders
            ))
            total_prl += len(prl_folders)

        from itertools import combinations

        # Each subject record:
        # (subid, prl_folders, lesion_folders)

        rng = np.random.default_rng(42)

        def counts(subject):
            _, prls, lesions = subject
            return len(prls), len(lesions)


        def subject_counts(subject):
            """Return counts in order: PRLs, subjects, non-PRL lesions."""
            return np.array([
                len(subject[1]),
                1,
                len(subject[2]),
            ], dtype=float)


        def imbalance_score(loads, targets, weights):
            """Weighted sum of squared relative deviations from targets."""
            return np.sum(
                weights * ((loads - targets) / targets)**2
            )


        # --------------------------------------------------
        # Stage 1: Select test subjects
        # --------------------------------------------------

        n_subjects = len(subject_folder_list)
        n_test_subjects = round(n_subjects * self.test_split)

        total_prls = sum(len(s[1]) for s in subject_folder_list)
        total_lesions = sum(len(s[2]) for s in subject_folder_list)

        target_test_prls = total_prls * self.test_split
        target_test_lesions = total_lesions * self.test_split

        # Exhaustive search is feasible for ~24 subjects.
        best_score = float("inf")
        best_indices = None

        for indices in combinations(range(n_subjects), n_test_subjects):
            test_prls = sum(len(subject_folder_list[i][1]) for i in indices)
            test_lesions = sum(len(subject_folder_list[i][2]) for i in indices)

            score = (
                3.0 * ((test_prls - target_test_prls) / max(target_test_prls, 1))**2
                + 1.0 * ((test_lesions - target_test_lesions) / max(target_test_lesions, 1))**2
            )

            if score < best_score:
                best_score = score
                best_indices = set(indices)

        test_fold_subjects = [
            s for i, s in enumerate(subject_folder_list)
            if i in best_indices
        ]

        train_subjects = [
            s for i, s in enumerate(subject_folder_list)
            if i not in best_indices
        ]


        # --------------------------------------------------
        # Stage 2: Assign training folds
        # --------------------------------------------------

        def assign_folds(
            subjects,
            n_folds,
            rng,
            weights=(3.0, 2.0, 1.0),
            n_restarts=200,
        ):
            weights = np.asarray(weights, dtype=float)

            counts = np.array([
                subject_counts(s) for s in subjects
            ])

            targets = np.maximum(counts.sum(axis=0) / n_folds, 1.0)

            best_score = np.inf
            best_folds = None

            for _ in range(n_restarts):

                ordered_subjects = list(subjects)
                rng.shuffle(ordered_subjects)
                ordered_subjects.sort(
                    key=lambda s: len(s[1]),
                    reverse=True,
                )

                folds = [[] for _ in range(n_folds)]
                loads = np.zeros((n_folds, 3))

                for subject in ordered_subjects:
                    contribution = subject_counts(subject)

                    # Evaluate the change in objective from adding
                    # this subject to each candidate fold.
                    scores = (
                        np.sum(
                            weights * (
                                (loads + contribution - targets) / targets
                            )**2,
                            axis=1,
                        )
                        - np.sum(
                            weights * (
                                (loads - targets) / targets
                            )**2,
                            axis=1,
                        )
                    )

                    k = int(np.argmin(scores))

                    folds[k].append(subject)
                    loads[k] += contribution

                score = imbalance_score(loads, targets, weights)

                if score < best_score:
                    best_score = score
                    best_folds = folds

            return {
                k: group for k, group in enumerate(best_folds)
            }


        weights = (3.0, 2.0, 1.0)

        fold_subjects = assign_folds(
            train_subjects,
            self.n_folds,
            rng,
            weights=weights,
        )

        def _make_entry(folder, subid, index, case_type, suffix=""):
            if len(suffix) > 0 and suffix[0] != "_":
                suffix = "_" + suffix
            rel = str(folder.relative_to(self.data_root))
            if case_type == "PRL":
                label = f"{rel}/{self.preprocess.prl_label_prefix}{suffix}_"
            else:
                label = f"{rel}/lesion_"
            return {
                "subid": subid, "lesion_index": index,
                "image": f"{rel}/",
                "label": label,
                "case_type": case_type,
            }

        datalist: dict[str, list] = {"training": [], "testing": []}
        label_types: dict[str, list] = {"prl_labels": [], "lesion_labels": []}

        for fold, ith_fold_subjects in fold_subjects.items():
            for subid, prl_folders, lesion_folders in ith_fold_subjects:
                for folder, index in prl_folders:
                    suffix = suffix_to_use.get(subid, "")
                    entry = _make_entry(
                        folder, subid, index, "PRL", suffix
                    )
                    entry['fold'] = fold
                    datalist['training'].append(entry)
                    label_types['prl_labels'].append(entry['label'])

                for folder, index in lesion_folders:
                    suffix = ""
                    entry = _make_entry(
                        folder, subid, index, "Lesion", suffix
                    )
                    entry['fold'] = fold
                    datalist['training'].append(entry)
                    label_types['lesion_labels'].append(entry['label'])


        for subid, prl_folders, lesion_folders in test_fold_subjects:
            for folder, index in prl_folders:
                suffix = suffix_to_use.get(subid, "")
                entry = _make_entry(
                    folder, subid, index, "PRL", suffix
                )
                entry['fold'] = fold
                datalist['testing'].append(entry)
                label_types['prl_labels'].append(entry['label'])
                
            for folder, index in lesion_folders:
                suffix = ""
                entry = _make_entry(
                    folder, subid, index, "Lesion", suffix
                )
                entry['fold'] = fold
                datalist['testing'].append(entry)
                label_types['lesion_labels'].append(entry['label'])
        
        
        with open(output_path, "w") as f:
            json.dump(datalist, f, indent=4)

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
        config = load_config(config_path, pwd=config_path.parent)

        # Resolve relative paths against dataset_home
        # TODO added this block before I added the ${PWD} token expansion to paths.py
        #   consider maybe just keeping this little block here for backwards compatibility but 
        #   making sure to always use ${PWD} to be explicit. OR I could update this block to 
        #   go through every key looking for values that look like they are relative path parts
        #   (e.g. anything with a file extension or anything with slashes), and resolving those
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