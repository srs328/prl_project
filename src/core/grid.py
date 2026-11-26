"""ExperimentGrid — HPO management via Cartesian product of parameters.

Generates run directories with per-run configs from a parameter grid,
and launches experiments locally or on HPC.
"""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import attrs
from functools import cached_property
from loguru import logger
import os

from core.configs import PreprocessingConfig, AlgoConfig
from core.dataset import Dataset
from core.experiment import Experiment
from helpers.parallel import BetterPool
from helpers.paths import PROJECT_ROOT


class ExperimentGrid:
    """Manages HPO across multiple Experiments via Cartesian product.

    The param_grid is a dict with optional "preprocessing" and "training" keys:

        param_grid:
          preprocessing:
            expand_xy: [10, 20, 30]
          training:
            learning_rate: [0.0001, 0.0002]
            crop_ratios: [null, [1, 1, 4]]

    When preprocessing params (expand_xy, expand_z) appear in the grid,
    generate() automatically calls dataset.create_rois() for each unique
    combination — fixing the gap in the old HPO scripts.
    """
    # TODO Make algo part of the grid definition then use that for fallback self.base_training
    def __init__(self, dataset: Dataset, param_grid: dict,
                 experiment_name: str,
                 base_preprocess: PreprocessingConfig | None = None,
                 base_training: AlgoConfig | None = None):
        self.dataset = dataset
        self.param_grid = param_grid
        self.experiment_name = experiment_name
        self.base_preprocess = base_preprocess or dataset.default_preprocess
        self.base_training = base_training or dataset.default_training
        self.work_home = dataset.work_home / experiment_name
        
    @property
    def runs(self) -> list[dict]:
        return self.run_params(
            self.param_grid.get("preprocessing", {}),
            self.param_grid.get("training", {})
        )
        
    @property
    def home(self) -> Path:
        """
        I'd prefer this name so I don't get confused with Dataset.work_home, 
        but right now I don't want to break anything by changing self.work_home
        """
        return self.work_home
    
    @cached_property
    def experiments(self) -> list[Experiment]:
        """Create Experiment objects for every run in the grid.

        Pure object creation — no filesystem side effects. Useful in notebooks
        or analysis code where you need Experiment handles without running
        preprocessing or setup.
        """
        experiments = []
        for entry in self.runs:
            pp_config = attrs.evolve(self.base_preprocess, **entry["preprocessing"])
            tr_config = attrs.evolve(self.base_training, **entry["training"])
            run_dir = self.work_home / entry["run_name"]
            exp = Experiment(
                dataset=self.dataset,
                preprocess_config=pp_config,
                training_config=tr_config,
                run_dir=run_dir,
            )
            experiments.append(exp)
        return experiments

    def generate(self, dry_run: bool = False, prepare_data: bool = True, validate: bool = False, overwrite: bool = False) -> list[Experiment]:
        """Generate all experiment run directories from param grid.

        Returns a list of Experiment objects. If expand_xy/expand_z/images
        are in the grid, creates ROIs and prepares data for each unique
        preprocessing config via a temporary Experiment.
        """
        experiments = self.experiments

        # Pre-create ROIs and datalists for all unique preprocessing configs
        if prepare_data and not dry_run:
            unique_pp_configs = {exp.preprocess_config for exp in experiments}
            for pp_config in unique_pp_configs:
                datalist_path = self.dataset.dataset_home / f"datalist_{pp_config.datalist_suffix}.json"
                if not datalist_path.exists():
                    logger.info(
                        f"Preparing data for {pp_config.datalist_suffix}..."
                    )
                    tmp_exp = Experiment(
                        dataset=self.dataset,
                        preprocess_config=pp_config,
                        training_config=self.base_training,
                        run_dir=Path("."),  # placeholder, not used
                    )
                    tmp_exp.create_rois()
                    self.dataset.create_datalist()
                    tmp_exp.prepare_data()

        manifest = {}
        for exp, entry in zip(experiments, self.runs):
            run_name = entry["run_name"]
            pp_dict = entry["preprocessing"]
            tr_dict = entry["training"]

            if dry_run:
                print(f"\n[DRY RUN] {run_name}:")
                if pp_dict:
                    print(f"  Preprocessing: {pp_dict}")
                print(f"  Training: {tr_dict}")
                print(f"  Run dir: {exp.run_dir}")
            else:
                import time
                t_run = time.perf_counter()
                exp.setup(validate=validate, overwrite=overwrite)
                elapsed = time.perf_counter() - t_run
                logger.debug(f"generate: {run_name} setup took {elapsed:.2f}s")
                print(f"Generated {run_name} in {exp.run_dir}")

            manifest[run_name] = {
                "preprocessing_params": pp_dict,
                "training_params": tr_dict,
                "run_dir": str(exp.run_dir),
            }

        # Write manifest
        if not dry_run:
            self.work_home.mkdir(parents=True, exist_ok=True)
            manifest_path = self.work_home / "runs_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Manifest saved to {manifest_path}")
            self.save_config()

        total = len(manifest)
        if dry_run:
            print(f"\n[DRY RUN] Would generate {total} total runs")
        else:
            print(f"\nTotal runs: {total}")

        return experiments

    def launch(self, experiments: list[Experiment] | None = None,
               mode: str = "local", processes: int = 1,
               dry_run: bool = False, run_key: str | None = None) -> None:
        """Launch experiments locally or on HPC.

        Args:
            experiments: List of experiments to launch. If None, reads from manifest.
            mode: "local" or "hpc".
            processes: Number of parallel processes for local mode.
            dry_run: Print commands without executing.
            run_key: Launch only a specific run (e.g., "run1").
        """
        if mode == "hpc":
            self._launch_hpc(dry_run=dry_run)
            return

        if experiments is None:
            experiments = self._load_experiments_from_manifest(run_key)

        if processes == 1:
            from tqdm import tqdm
            for exp in tqdm(experiments):
                self._launch_single(exp, dry_run=dry_run)
        else:
            from tqdm import tqdm
            tasks = [{"exp": exp, "dry_run": dry_run} for exp in experiments]
            with BetterPool(processes) as pool:
                results = pool.imap_unordered(
                    lambda t: self._launch_single(t["exp"], t["dry_run"]),
                    tasks,
                )
                for _ in tqdm(results, total=len(tasks)):
                    pass

    def _launch_single(self, exp: Experiment, dry_run: bool = False) -> None:
        """Launch a single experiment via subprocess."""
        # Use the train.py in the dataset's dataset_home
        train_script = self.dataset.dataset_home / "train.py"

        cmd = f"cd {exp.run_dir} && {sys.executable} {train_script} --run-dir {exp.run_dir}"

        if dry_run:
            print(f"[DRY RUN] {exp.run_dir.name}: {cmd}")
        else:
            logger.info(f"Launching {exp.run_dir.name}...")
            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"  {exp.run_dir.name} completed")
            except subprocess.CalledProcessError as e:
                print(f"  {exp.run_dir.name} FAILED: {e}")
                raise

    def _launch_hpc(self, dry_run: bool = False) -> None:
        """Generate and submit LSF job array for HPC."""
        manifest_path = self.work_home / "runs_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run generate() first."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        runs = list(manifest.values())
        train_script = self.dataset.dataset_home / "train.py"

        script_content = f"""#!/bin/bash
#BSUB -J "prl_hpo[1-{len(runs)}]"
#BSUB -n 3
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:j_exclusive=yes"
#BSUB -R "select[hname!='c4140c08'] rusage[mem=4G]"
#BSUB -W 24:00
#BSUB -o {self.work_home}/logs/run_%J_%I.log
#BSUB -e {self.work_home}/logs/run_%J_%I.err

set -euo pipefail
source {PROJECT_ROOT}/hpc/setup_env_hpc.sh

echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array index: $LSB_JOBINDEX"

runs=(
"""
        for run_info in runs:
            script_content += f'    "{run_info["run_dir"]}"\n'

        script_content += f""")

run_dir=${{runs[$LSB_JOBINDEX - 1]}}

echo "Starting training in $run_dir"
cd "$run_dir"
{sys.executable} {train_script} --run-dir "$run_dir"
echo "Training in $run_dir completed with exit code $?"
"""

        submit_script = self.work_home / "submit_array.sh"
        submit_script.write_text(script_content)
        submit_script.chmod(0o755)

        logs_dir = self.work_home / "logs"
        logs_dir.mkdir(exist_ok=True)

        if dry_run:
            print(f"[DRY RUN] Would submit: bsub < {submit_script}")
        else:
            try:
                subprocess.run(f"bsub < {submit_script}", shell=True, check=True)
                print(f"Submitted {len(runs)} jobs to HPC")
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit to HPC: {e}")
                raise

    def _load_experiments_from_manifest(self, run_key: str | None = None) -> list[Experiment]:
        """Load experiments from the manifest file."""
        manifest_path = self.work_home / "runs_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run generate() first."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        if run_key:
            entries = {run_key: manifest[run_key]}
        else:
            entries = manifest

        experiments = []
        for run_name, info in entries.items():
            run_dir = Path(info["run_dir"])
            exp = Experiment.from_run_dir(run_dir, self.dataset)
            experiments.append(exp)

        return experiments

    def save_config(self, output_path: Path | None = None) -> Path:
        """Save the grid configuration to a YAML file.

        Produces a file that can be passed back to from_config() to recreate
        this grid exactly. Useful when the grid was constructed by direct
        initialization rather than loaded from a file.

        Args:
            output_path: Destination file. Defaults to
                work_home/experiment_config.yaml.

        Returns:
            The path the config was written to.
        """
        import yaml

        if output_path is None:
            self.work_home.mkdir(parents=True, exist_ok=True)
            output_path = self.work_home / "experiment_config.yaml"

        config: dict = {
            "dataset_name": self.dataset.name,
            "experiment_name": self.experiment_name,
            "param_grid": self.param_grid,
        }

        # Include base config overrides only when they differ from dataset defaults.
        # This keeps the saved file clean for the common case.
        default_pp = self.dataset.default_preprocess
        if self.base_preprocess != default_pp:
            config["base_preprocessing"] = attrs.asdict(self.base_preprocess)

        default_tr = self.dataset.default_training
        if attrs.asdict(self.base_training) != attrs.asdict(default_tr):
            config["base_training"] = attrs.asdict(self.base_training)

        output_path = Path(output_path)
        output_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        logger.info(f"Experiment config saved to {output_path}")
        return output_path

    @classmethod
    def from_config(cls, config_path: Path) -> ExperimentGrid:
        """Create an ExperimentGrid from an experiment config YAML/JSON file.

        Expected format:
            dataset_name: roi_train2
            experiment_name: stage1_crop_lr_sweep
            param_grid:
              training:
                crop_ratios: [null, [1, 1, 4]]
                learning_rate: [0.0001, 0.0002]

        Optional base config overrides (produced by save_config when they
        differ from dataset defaults):
            base_preprocessing:
              expand_xy: 10
            base_training:
              learning_rate: 0.0001
        """
        from helpers.paths import load_config

        config = load_config(config_path)
        dataset = Dataset(config["dataset_name"])

        base_preprocess = None
        if "base_preprocessing" in config:
            base_preprocess = attrs.evolve(
                dataset.default_preprocess, **config["base_preprocessing"]
            )

        base_training = None
        if "base_training" in config:
            base_training = AlgoConfig.from_dict(config["base_training"])

        return cls(
            dataset=dataset,
            param_grid=config["param_grid"],
            experiment_name=config["experiment_name"],
            base_preprocess=base_preprocess,
            base_training=base_training,
        )
        
    @classmethod
    def from_home_dir(cls, work_home: os.PathLike):
        config_path = Path(work_home) / "experiment_config.yaml"
        return cls.from_config(config_path)
    
    def get_info(self) -> tuple[dict[str], dict[str]]:
        """Return string identifiers of each hyperparameter value
        
        Returns a tuple containing lists of strings for the preprocessing 
        and training parameters that were varied. If string identifiers were 
        specified in the config, these will be returned. Otherwise, the actual 
        parameters will be returned in string format
        """
        preprocess_params = self.param_grid.get("preprocessing", {}) 
        preprocess_param_ids = self.param_grid.get("preprocessing_ids", preprocess_params)
        preprocess_param_ids = {
            k: [str(item) for item in v] for k, v in preprocess_param_ids.items()
        }
        
        training_params = self.param_grid.get("training", {})
        training_param_ids = self.param_grid.get("training_ids", training_params)
        training_param_ids = {
            k: [str(item) for item in v] for k, v in training_param_ids.items()
        }
        
        return preprocess_param_ids, training_param_ids

    @staticmethod
    def run_params(preprocess_params: dict, training_params: dict) -> list[dict]:
        """Return a run-name → parameter mapping for the given param dicts.

        Each entry has keys: 'run_name', 'preprocessing', 'training'.
        Order is deterministic and matches the directories generate() creates.
        No side effects — safe to call in notebooks without touching the filesystem.

        Pass raw param dicts for actual values, or string-ID dicts from get_info()
        for human-readable labels::

            # actual values
            pp = eg.param_grid.get('preprocessing', {})
            tr = eg.param_grid.get('training', {})
            df = pd.DataFrame(eg.run_params(pp, tr))

            # string labels (from get_info)
            pp_ids, tr_ids = eg.get_info()
            df = pd.DataFrame(eg.run_params(pp_ids, tr_ids))
        """
        pp_keys = list(preprocess_params.keys())
        pp_combos = list(product(*[preprocess_params[k] for k in pp_keys])) if pp_keys else [()]

        tr_keys = list(training_params.keys())
        tr_combos = list(product(*[training_params[k] for k in tr_keys])) if tr_keys else [()]

        runs = []
        run_num = 1
        for pp_combo in pp_combos:
            for tr_combo in tr_combos:
                runs.append({
                    "run_name": f"run{run_num}",
                    "preprocessing": dict(zip(pp_keys, pp_combo)),
                    "training": dict(zip(tr_keys, tr_combo)),
                })
                run_num += 1
        return runs

    def __repr__(self) -> str:
        return (
            f"ExperimentGrid(dataset={self.dataset.name}, "
            f"experiment={self.experiment_name}, "
            f"work_home={self.work_home})"
        )