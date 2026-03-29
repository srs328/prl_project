"""Experiment class — a single training run.

Encapsulates setup, training (via MONAI AutoRunner), prediction, and
preprocessing for a single set of hyperparameters. Case path resolution
is delegated to Dataset; Experiment adds inference output paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from functools import cached_property

import pandas as pd
from loguru import logger

from core.configs import PreprocessingConfig, AlgoConfig
from core.dataset import Dataset


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
        self.preprocess_config = preprocess_config
        self.training_config = training_config
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_absolute():
            self.run_dir = dataset.work_home / self.run_dir

        # Ensure dataset uses this experiment's preprocessing config for
        # path resolution. If the dataset already has the right config
        # (or no override), set it; otherwise create a view with the
        # correct config.
        if dataset._preprocess is None or dataset._preprocess == preprocess_config:
            dataset._preprocess = preprocess_config
            self.dataset = dataset
        else:
            # Different preprocess config — create a new Dataset instance
            # that shares the same identity but resolves paths differently
            self.dataset = Dataset(dataset.name, preprocess=preprocess_config)

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
        """Returns the actual hyper_params used in training."""
        for fold_n in range(self.dataset.n_folds):
            fold_dir = self.run_dir / f"{self.training_config.algo}_{fold_n}"
            if self.has_trained(fold_dir):
                hyper_params_file = fold_dir / "configs/hyper_parameters.yaml"
                params = AlgoConfig.load_from_yaml(hyper_params_file)
                return params

    @property
    def work_home(self) -> Path:
        """Alias for run_dir (consistent with Dataset/ExperimentGrid naming)."""
        return self.run_dir

    @property
    def name(self) -> str:
        return str(self.run_dir.relative_to(self.dataset.work_home))

    @property
    def id(self) -> str:
        return str(self.run_dir.relative_to(self.dataset.work_home.parent))

    # --- Preprocessing ---

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

    @cached_property
    def cases(self) -> pd.DataFrame:
        """Dataset cases augmented with inference paths for this run.

        Returns a DataFrame indexed by (subid, lesion_index) with columns:
        split, case_type, image, label, subject_dir, inference.

        The 'inference' column contains the path to the model's prediction
        output if it exists on disk, or None otherwise.

        Call refresh_cases() to re-scan disk after predict() generates
        new inference files.
        """
        df = self.dataset.cases.copy()
        df["inference"] = df.apply(self._find_inference_path, axis=1)
        return df

    def refresh_cases(self) -> None:
        """Invalidate cached cases so the next access re-scans disk."""
        # Clear the cached_property
        self.__dict__.pop("cases", None)
        # Also clear dataset's cached cases so paths are re-resolved
        self.dataset.__dict__.pop("cases", None)

    def _find_inference_path(self, row: pd.Series) -> Path | None:
        """Resolve inference output path for a single case row."""
        data_root = self.dataset.data_root
        image = row["image"]
        label = row["label"]
        split = row.name[0] if isinstance(row.name, tuple) else row["split"]
        # row.name is the index tuple (subid, lesion_index), split is a column
        split = row["split"]

        label_relative = label.relative_to(data_root)

        if split == "testing":
            inf_path = (
                self.run_dir
                / "ensemble_output"
                / label_relative.with_name(
                    f"{image.name.removesuffix('.nii.gz')}_ensemble.nii.gz"
                )
            )
        else:
            inf_path = (
                self.run_dir
                / "fold_predictions"
                / split
                / label_relative.with_name(image.name)
            )

        return inf_path if inf_path.exists() else None

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
