"""Experiment class — a single training run.

Encapsulates setup, training (via MONAI AutoRunner), prediction, and
evaluation for a single set of hyperparameters. Also owns ROI creation
and data preparation, since these depend on per-experiment parameters
(expand_xy, expand_z, images).
"""

from __future__ import annotations

import json
import os
import sys
import contextlib
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from functools import cached_property
from typing import TYPE_CHECKING

from loguru import logger

from core.configs import PreprocessingConfig, AlgoConfig
from core.dataset import Dataset

if TYPE_CHECKING:
    import pandas as pd
    
class Experiment:
    """A single training run with fixed hyperparameters.

    Owns a reference to a Dataset, preprocessing config, training config,
    and a run directory where all outputs live.
    """

    def __init__(
        self,
        dataset: Dataset,
        preprocess_config: PreprocessingConfig,
        training_config: AlgoConfig,
        run_dir: Path,
    ):
        self.dataset = dataset
        self.preprocess_config = preprocess_config
        self.training_config = training_config
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_absolute():
            self.run_dir = dataset.work_home / self.run_dir
        self._cases: list[dict] | None = None

    @property
    def datalist_name(self) -> str:
        return f"datalist_{self.preprocess_config.datalist_suffix}.json"

    @property
    def datalist_src(self) -> Path:
        """Path to the datalist in the dataset's dataset_home."""
        return self.dataset.dataset_home / self.datalist_name

    @property
    def datalist_dst(self) -> Path:
        """Path to the datalist copy in the run directory."""
        return self.run_dir / self.datalist_name

    @property
    def hyper_params(self) -> AlgoConfig:
        """Returns the actual hyper_params used in training"""
        for fold_n in range(self.dataset.n_folds):
            # TODO Make this a function if other algos name folds differently 
            fold_dir = self.run_dir / f"{self.training_config.algo}_{fold_n}"
            if self.has_trained(fold_dir):
                hyper_params_file = fold_dir / "configs/hyper_parameters.yaml"
                params = AlgoConfig.load_from_yaml(hyper_params_file)
                return params
    
    @property
    def work_home(self) -> Path:
        """To be consistent with naming pattern used in Dataset and ExperimentGrid"""
        return self.run_dir
    
    @property
    def name(self) -> str:
        return str(self.run_dir.relative_to(self.dataset.work_home))
    
    @property
    def id(self) -> str:
        return str(self.run_dir.relative_to(self.dataset.work_home.parent))
    
    @cached_property
    def cases_df(self) -> pd.DataFrame:
        import pandas as pd
        return pd.DataFrame(self.cases).set_index(["subid", "lesion_index"])
    
    def get_case(self, subid, lesion_index) -> dict:
        iloc = self.cases_df.index.get_loc((subid, lesion_index))
        return self.cases_df.reset_index().iloc[iloc].to_dict()


    # --- Preprocessing (moved from Dataset) ---

    def create_rois(self) -> None:
        """Crop ROIs for all subjects at this experiment's expand_xy/expand_z."""
        from preprocessing.create_rois import create_rois_for_subjects

        cfg = self.preprocess_config
        create_rois_for_subjects(
            subjects=self.dataset.subjects,
            suffix_to_use=self.dataset.suffix_to_use,
            prl_df=self.dataset.prl_df,
            data_root=self.dataset.data_root,
            expand_xy=cfg.expand_xy,
            expand_z=cfg.expand_z,
            processes=cfg.processes,
            dry_run=cfg.dry_run,
        )

    def prepare_data(self) -> Path:
        """Stack channels and produce the final datalist with expansion suffixes."""
        from preprocessing.prepare_training_data import prepare_training_data

        cfg = self.preprocess_config
        return prepare_training_data(
            datalist_template_path=self.dataset.datalist_template_path,
            data_root=self.dataset.data_root,
            images=cfg.images,
            expand_xy=cfg.expand_xy,
            expand_z=cfg.expand_z,
            output_path=self.datalist_src,
        )

    # --- Datalist & cases ---

    @property
    def datalist(self) -> dict:
        """Load the prepared datalist (from run_dir if available, else source_home)."""
        path = self.datalist_dst if self.datalist_dst.exists() else self.datalist_src
        with open(path) as f:
            return json.load(f)

    @property
    def cases(self) -> list[dict]:
        """Flat list of all cases with split, case_type, and inference paths.

        Cached after first access. Call refresh_cases() to re-scan disk
        (e.g. after predict() generates new inference files).
        """
        if self._cases is None:
            self._cases = self._build_cases()
        return self._cases
    
    def subject_dir(self, subid):
        return self.datalist

    def refresh_cases(self) -> None:
        """Invalidate cached cases so the next access re-scans disk."""
        self._cases = None

    def _build_cases(self) -> list[dict]:
        datalist = self.datalist
        cases = []

        for item in datalist.get("testing", []):
            item = dict(item)
            item["split"] = "testing"
            self._resolve_inference_path(item)
            resolve_case_type(item)
            cases.append(item)

        for item in datalist.get("training", []):
            item = dict(item)
            item["split"] = f"fold{item.pop('fold')}"
            self._resolve_inference_path(item)
            resolve_case_type(item)
            cases.append(item)

        return cases

    def _resolve_inference_path(self, case: dict) -> None:
        """Add 'inference' key to a case dict if the predicted file exists.

        Also converts image/label to absolute Paths.
        """
        data_root = self.dataset.data_root
        cfg = self.preprocess_config

        # Make label absolute
        label = Path(case["label"])
        if not label.is_absolute():
            label = data_root / label
        case["label"] = label

        # Make image absolute
        image = Path(case["image"])
        if not image.is_absolute():
            image = data_root / image
        case["image"] = image

        # Resolve inference output path
        label_relative = label.relative_to(data_root)
        if case["split"] == "testing":
            # inf_path = (
            #     self.run_dir
            #     / "ensemble_output"
            #     / label_relative.with_name(f"{cfg.datalist_suffix}_ensemble.nii.gz")
            # )
            inf_path = (
                self.run_dir
                / "ensemble_output"
                / label_relative.with_name(f"{image.name.removesuffix('.nii.gz')}_ensemble.nii.gz")
            )
        else:
            # Validation: fold_predictions/fold0/...
            # inf_path = (
            #     self.run_dir
            #     / "fold_predictions"
            #     / case["split"]
            #     / label_relative.with_name(f"{cfg.datalist_suffix}.nii.gz")
            # )
            inf_path = (
                self.run_dir
                / "fold_predictions"
                / case["split"]
                / label_relative.with_name(f"{image.name}")
            )

        if inf_path.exists():
            case["inference"] = inf_path

    # --- Setup ---

    def setup(self, validate: bool = True, overwrite: bool = False) -> None:
        """Create run directory and write configs + datalist into it.

        Args:
            validate: Check that every image/label path in the datalist exists
                on disk. Set to False when generating many runs in a grid where
                the datalist has already been validated — avoids repeated SMB
                round-trips that can add several seconds per run.
            overwrite: If False (default), skip setup silently if label_config.json
                already exists. Set to True to re-write all config files.
        """
        import time

        t0 = time.perf_counter()

        if not overwrite and (self.run_dir / "label_config.json").exists():
            logger.warning(
                f"setup [{self.run_dir.name}]: already set up, skipping (pass overwrite=True to force)"
            )
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"setup [{self.run_dir.name}]: mkdir done ({time.perf_counter() - t0:.2f}s)"
        )

        # Ensure datalist exists (create ROIs + prepare data if needed)
        if not self.datalist_src.exists():
            logger.debug(
                f"setup [{self.run_dir.name}]: datalist_src missing, running preprocessing"
            )
            self.create_rois()
            self.dataset.create_datalist()
            self.prepare_data()
        logger.debug(
            f"setup [{self.run_dir.name}]: datalist check done ({time.perf_counter() - t0:.2f}s)"
        )

        # Write label_config.json
        label_cfg = self.training_config.to_label_config_dict(
            self.preprocess_config, self.dataset
        )
        with open(self.run_dir / "label_config.json", "w") as f:
            json.dump(label_cfg, f, indent=2)
        logger.debug(
            f"setup [{self.run_dir.name}]: label_config written ({time.perf_counter() - t0:.2f}s)"
        )

        # Write monai_config.json
        monai_cfg = self.training_config.to_monai_config_dict(self.dataset)
        with open(self.run_dir / "monai_config.json", "w") as f:
            json.dump(monai_cfg, f, indent=2)
        logger.debug(
            f"setup [{self.run_dir.name}]: monai_config written ({time.perf_counter() - t0:.2f}s)"
        )

        # Copy datalist
        if not self.datalist_dst.exists():
            import shutil

            shutil.copyfile(self.datalist_src, self.datalist_dst)
        logger.debug(
            f"setup [{self.run_dir.name}]: datalist copied ({time.perf_counter() - t0:.2f}s)"
        )

        # Validate image/label paths exist (skippable for grid generation)
        if validate:
            datalist = self.datalist
            n = len(datalist.get("training", [])) + len(datalist.get("testing", []))
            logger.debug(
                f"setup [{self.run_dir.name}]: validating {n} cases against data_root (SMB)..."
            )
            for item in datalist.get("training", []) + datalist.get("testing", []):
                img_path = self.dataset.data_root / item["image"]
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                img_path = self.dataset.data_root / item["label"]
                if not img_path.exists():
                    raise FileNotFoundError(f"Label not found: {img_path}")
            logger.debug(
                f"setup [{self.run_dir.name}]: validation done ({time.perf_counter() - t0:.2f}s)"
            )

        # Write run info
        cfg = self.preprocess_config
        description = (
            f"Training run\n"
            f"dataset={self.dataset.name}\n"
            f"images={list(cfg.images)}\n"
            f"expand_xy={cfg.expand_xy}, expand_z={cfg.expand_z}\n"
            f"run_dir={self.run_dir}\n"
            f"learning_rate={self.training_config.learning_rate}\n"
            f"num_epochs={self.training_config.num_epochs}\n"
        )
        with open(self.run_dir / "info.txt", "w") as f:
            f.write(description)

        logger.info(
            f"Experiment setup complete: {self.run_dir} ({time.perf_counter() - t0:.2f}s)"
        )

    # --- Training ---

    def train(self) -> None:
        """Run MONAI AutoRunner training."""
        from monai.apps.auto3dseg import AutoRunner

        if not self.datalist_dst.exists():
            self.setup()

        # All params flow through the input dict → fill_template_config() →
        # hyper_parameters.yaml. No set_training_params() needed.
        input_dict = self.training_config.to_input_dict(
            self.datalist_dst, self.dataset.data_root
        )

        # MLflow setup
        mlflow_tracking_uri = str(self.run_dir / "mlruns")
        run_name = self.run_dir.name
        mlflow_experiment_name = (
            run_name[3:] if run_name.startswith("run") else run_name
        )
        runner = AutoRunner(
            work_dir=self.run_dir,
            algos=[self.training_config.algo],
            input=input_dict,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
        )

        logger.info(f"Starting training in {self.run_dir}")
        try:
            runner.run()
        except Exception:
            self.cleanup(self.run_dir, success=False)
            raise
        else:
            self.cleanup(self.run_dir, success=True)

    @staticmethod
    def cleanup(run_dir, success=True):
        import shutil
        # FIXME Should use algo to be generic (see hyper_param property)
        for fold_dir in sorted(run_dir.glob("segresnet_*")):
            log_path: Path = fold_dir / "model/training.log"
            if success:
                if log_path.exists():
                    shutil.copy(log_path, log_path.parent / "orig_training.log")
            else:
                # Clean up incomplete folds so AutoRunner doesn't skip them.
                # When a job is killed mid-training, algo_object.pkl exists (from algo_gen)
                # with best_metric=None, but progress.yaml has partial scores that fool
                # AutoRunner's get_score() into thinking the fold completed.
                if not Experiment.has_trained(fold_dir):
                    progress_path = fold_dir / "model" / "progress.yaml"
                    logger.info(
                        f"Incomplete fold detected: {fold_dir.name} "
                        "— removing progress.yaml"
                    )
                    progress_path.unlink()
                    # delete the training log too
                    log_path.unlink(missing_ok=True)

    @staticmethod
    def has_trained(fold_dir):
        import pickle

        pkl_path = fold_dir / "algo_object.pkl"
        progress_path = fold_dir / "model" / "progress.yaml"
        if pkl_path.exists() and progress_path.exists():
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            if pkl_data.get("best_metric") is None:
                return False
        return True

    # --- Prediction ---

    def predict(
        self, fold: int | None = None, regenerate: bool = False
    ) -> dict[int, str]:
        """Run fold validation inference.

        Args:
            fold: Specific fold number, or None for all folds.
            regenerate: Re-run even if outputs already exist. Default False.

        Returns:
            Dict mapping fold number to "success" or error message.
        """
        from scripts.generate_fold_predictions import run_fold_inference

        output_dir = self.run_dir / "fold_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        datalist = self.datalist

        if fold is not None:
            folds = [fold]
        else:
            fold_nums = set(item.get("fold") for item in datalist.get("training", []))
            folds = sorted(fold_nums)

        results = {}
        for fold_num in folds:
            try:
                success = run_fold_inference(
                    self.run_dir,
                    fold_num,
                    self.datalist_dst,
                    self.dataset.data_root,
                    output_dir,
                    regenerate=regenerate,
                )
                results[fold_num] = "success" if success else "failed"
            except Exception as e:
                logger.error(f"Error processing fold {fold_num}: {e}")
                results[fold_num] = f"error: {e}"

        self.refresh_cases()
        return results

    # --- Evaluation ---
    # TODO Consider removing from inside class
    def evaluate(
        self,
        test_only: bool = False,
        output_csv: Path | None = None,
        print_results: bool | str = False,
    ) -> pd.DataFrame | None:
        """Compute performance metrics using self.cases.

        Groups cases by split and runs analyze_dataset() on each group.
        Returns a DataFrame of per-case metrics.
        """
        from scripts.compute_performance_metrics import (
            analyze_dataset,
            print_results as _print_results,
        )
        import pandas as _pd

        cases = self.cases

        # Group by split
        by_split: dict[str, list[dict]] = defaultdict(list)
        for case in cases:
            if "inference" not in case:
                continue
            by_split[case["split"]].append(case)

        all_results = {}

        # Process testing first
        if "testing" in by_split:
            results = analyze_dataset(by_split["testing"], split="testing")
            if results.get("aggregated"):
                all_results["testing"] = results
                if print_results:
                    if type(print_results) is str:
                        with open(print_results, "a") as f:
                            with contextlib.redirect_stdout(f):
                                _print_results(results)
                    else:
                        _print_results(results)

        # Process validation folds
        if not test_only:
            for split_name in sorted(by_split):
                if split_name == "testing":
                    continue
                results = analyze_dataset(
                    by_split[split_name], split=f"validation {split_name}"
                )
                if results.get("aggregated"):
                    all_results[f"validation {split_name}"] = results
                    if print_results:
                        _print_results(results)

        # Build DataFrame
        all_cases = []
        for split, results in all_results.items():
            for case in results["cases"]:
                case["split"] = split
                all_cases.append(case)

        if not all_cases:
            return None

        df = _pd.DataFrame(all_cases)
        if output_csv:
            df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to: {output_csv}")

        return df

    # --- Class methods ---

    @classmethod
    def from_run_dir(cls, run_dir: Path, dataset: Dataset | None = None) -> Experiment:
        """Reconstruct an Experiment from an existing run directory.

        Reads label_config.json and monai_config.json from the run_dir
        to reconstruct the configs.
        """
        from helpers.paths import load_config
        from core.configs import PreprocessingConfig, AlgoConfig

        run_dir = Path(run_dir)
        if not run_dir.is_absolute():
            if dataset is None:
                raise ValueError(
                    "run_dir must be an absolute path if dataset is not provided."
                )    
            run_dir = dataset.work_home / run_dir

        label_config = load_config(run_dir / "label_config.json")
        monai_config = load_config(run_dir / "monai_config.json")
        
        preprocess_config = PreprocessingConfig(
            expand_xy=label_config["expand_xy"],
            expand_z=label_config["expand_z"],
            images=label_config.get("images", ["flair", "phase"]),
        )

        train_param = monai_config.get("train_param", {})
        training_config = AlgoConfig.from_dict(train_param)
        
        if dataset is None:
            dataset = Dataset(label_config['dataset_name'])
        
        return cls(
            dataset=dataset,
            preprocess_config=preprocess_config,
            training_config=training_config,
            run_dir=run_dir,
        )

    def next_run_dir(self) -> Path:
        """Auto-increment to next available run<N> directory under work_home."""
        run_num = 1
        while (self.dataset.work_home / f"run{run_num}").exists():
            run_num += 1
        return self.dataset.work_home / f"run{run_num}"

    def __repr__(self) -> str:
        return f"Experiment(id={self.id}, run_dir={self.run_dir}, dataset={self.dataset.name})"


def resolve_case_type(t_case):
    if "prl" in Path(t_case["label"]).name:
        t_case["case_type"] = "PRL"
    else:
        t_case["case_type"] = "Lesion"
