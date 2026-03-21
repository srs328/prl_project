"""Experiment class — a single training run.

Encapsulates setup, training (via MONAI AutoRunner), prediction, and
evaluation for a single set of hyperparameters.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import pandas as pd
    from core.dataset import Dataset
    from core.configs import PreprocessingConfig, TrainingConfig


class Experiment:
    """A single training run with fixed hyperparameters.

    Owns a reference to a Dataset, preprocessing config, training config,
    and a run directory where all outputs live.
    """

    def __init__(self, dataset: Dataset, preprocess_config: PreprocessingConfig,
                 training_config: TrainingConfig, run_dir: Path):
        self.dataset = dataset
        self.preprocess_config = preprocess_config
        self.training_config = training_config
        self.run_dir = Path(run_dir)

    @property
    def datalist_name(self) -> str:
        return f"datalist_{self.preprocess_config.suffix}.json"

    @property
    def datalist_src(self) -> Path:
        """Path to the datalist in the dataset's source_home."""
        return self.dataset.datalist_path(self.preprocess_config)

    @property
    def datalist_dst(self) -> Path:
        """Path to the datalist copy in the run directory."""
        return self.run_dir / self.datalist_name

    def setup(self) -> None:
        """Create run directory and write configs + datalist into it.

        Replaces the manual config copying in the old train.py.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Write label_config.json
        label_cfg = self.training_config.to_label_config_dict(
            self.preprocess_config, self.dataset
        )
        with open(self.run_dir / "label_config.json", "w") as f:
            json.dump(label_cfg, f, indent=2)

        # Write monai_config.json
        monai_cfg = self.training_config.to_monai_config_dict(self.dataset)
        with open(self.run_dir / "monai_config.json", "w") as f:
            json.dump(monai_cfg, f, indent=2)

        # Copy datalist
        if not self.datalist_dst.exists():
            if not self.datalist_src.exists():
                raise FileNotFoundError(
                    f"Source datalist not found: {self.datalist_src}. "
                    f"Run dataset.prepare_data() first."
                )
            import shutil
            shutil.copyfile(self.datalist_src, self.datalist_dst)

        # Validate image/label paths exist
        with open(self.datalist_dst) as f:
            datalist = json.load(f)

        dataroot = self.dataset.data_root
        for item in datalist.get("training", []) + datalist.get("testing", []):
            img_path = Path(dataroot) / item["image"]
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

        # Write run info
        description = (
            f"Training run\n"
            f"dataset={self.dataset.name}\n"
            f"expand_xy={self.preprocess_config.expand_xy}, "
            f"expand_z={self.preprocess_config.expand_z}\n"
            f"run_dir={self.run_dir}\n"
            f"learning_rate={self.training_config.learning_rate}\n"
            f"num_epochs={self.training_config.num_epochs}\n"
        )
        with open(self.run_dir / "info.txt", "w") as f:
            f.write(description)

        logger.info(f"Experiment setup complete: {self.run_dir}")

    def train(self) -> None:
        """Run MONAI AutoRunner training.

        This replaces the logic from training/roi_train2/train.py.
        """
        from monai.apps.auto3dseg import AutoRunner

        if not self.datalist_dst.exists():
            self.setup()

        # Build train_param (excludes list-valued params)
        train_param = self.training_config.to_train_param()
        algos = self.training_config.algos

        # Remove list-valued params from train_param (they go into input_dict)
        for key in list(train_param):
            if isinstance(train_param[key], list):
                del train_param[key]

        # Build input dict
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
            algos=algos,
            input=input_dict,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
        )
        runner.set_training_params(train_param)

        logger.info(f"Starting training in {self.run_dir}")
        runner.run()

    def predict(self, fold: int | None = None) -> dict[int, str]:
        """Run fold validation inference.

        Uses the Auto3DSeg inference infrastructure in each fold's scripts/ dir.

        Args:
            fold: Specific fold number, or None for all folds.

        Returns:
            Dict mapping fold number to "success" or error message.
        """
        from scripts.generate_fold_predictions import (
            run_fold_inference, get_validation_cases,
        )

        output_dir = self.run_dir / "fold_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.datalist_dst) as f:
            datalist = json.load(f)

        if fold is not None:
            folds = [fold]
        else:
            fold_nums = set(
                item.get("fold") for item in datalist.get("training", [])
            )
            folds = sorted(fold_nums)

        results = {}
        for fold_num in folds:
            try:
                success = run_fold_inference(
                    self.run_dir, fold_num, self.datalist_dst,
                    self.dataset.data_root, output_dir,
                )
                results[fold_num] = "success" if success else "failed"
            except Exception as e:
                logger.error(f"Error processing fold {fold_num}: {e}")
                results[fold_num] = f"error: {e}"

        return results

    def evaluate(self, test_only: bool = False, output_csv: Path | None = None,
                 print_results: bool = False) -> pd.DataFrame | None:
        """Compute performance metrics.

        Wraps compute_performance_metrics.py logic. Returns a DataFrame
        of per-case metrics if output_csv is requested.
        """
        from scripts.compute_performance_metrics import (
            get_test_inference, get_validation_inference,
            analyze_dataset, print_results as _print_results,
        )
        import pandas as _pd

        label_config_path = self.run_dir / "label_config.json"
        if not label_config_path.exists():
            raise FileNotFoundError(
                f"label_config.json not found in {self.run_dir}. Run setup() first."
            )

        from helpers.paths import load_config
        label_config = load_config(label_config_path)
        expand_xy, expand_z = label_config["expand_xy"], label_config["expand_z"]
        expand_suffix = f"_xy{expand_xy}_z{expand_z}"

        datalist_path = self.run_dir / f"datalist_xy{expand_xy}_z{expand_z}.json"
        with open(datalist_path) as f:
            datalist = json.load(f)

        dataroot = label_config["dataroot"]
        inf_root = self.run_dir / "ensemble_output"
        val_root = self.run_dir / "fold_predictions"

        all_results = {}

        inf_data = get_test_inference(
            datalist["testing"], dataroot, inf_root, expand_suffix
        )
        inf_results = analyze_dataset(inf_data, split="testing")
        if inf_results.get("aggregated"):
            all_results["testing"] = inf_results
            if print_results:
                _print_results(inf_results)

        if not test_only:
            val_data = get_validation_inference(
                datalist["training"], dataroot, val_root, expand_suffix
            )
            for fold_name, fold_data in val_data.items():
                split = f"validation {fold_name}"
                results = analyze_dataset(fold_data, split=split)
                if results.get("aggregated"):
                    all_results[split] = results
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

    @classmethod
    def from_run_dir(cls, run_dir: Path, dataset: Dataset) -> Experiment:
        """Reconstruct an Experiment from an existing run directory.

        Reads label_config.json and monai_config.json from the run_dir
        to reconstruct the configs.
        """
        from helpers.paths import load_config
        from core.configs import PreprocessingConfig, TrainingConfig

        run_dir = Path(run_dir)
        label_config = load_config(run_dir / "label_config.json")
        monai_config = load_config(run_dir / "monai_config.json")

        preprocess_config = PreprocessingConfig(
            expand_xy=label_config["expand_xy"],
            expand_z=label_config["expand_z"],
        )

        train_param = monai_config.get("train_param", {})
        training_config = TrainingConfig.from_dict(train_param)

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
        return f"Experiment(run_dir={self.run_dir}, dataset={self.dataset.name})"
