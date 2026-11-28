# PRL pipeline setup

The checkout installs the Python package `prl` and the command `prl`. From
the checkout root, activate the environment you use for this project and install
the package in editable mode.

```bash
source ~/.virtualenvs/monai/bin/activate
python -m pip install -e .
prl --help
```

Use the same interpreter for VS Code and notebook kernels. Keep this checkout
and `prl_pipeline-v2` in separate environments because both distributions are
named `prl-pipeline`. The HPC jobs use the checkout's `.venv`. Install the
checkout in that environment before submitting jobs.

Dataset configs, resources, and training templates come from the checkout. Set
`PRL_PROJECT_ROOT` to its absolute path when running a wheel installation from
elsewhere. Set `PRL_DATA_ROOT` and `PRL_TRAIN_ROOT` to override the default data
and training locations.

## MLflow commands

```bash
# Collate a distributed run.
python -m prl.scripts.mlflow_cli collate /path/to/run2

# Build a combined view using resources/experiment_runs.txt.
python -m prl.scripts.mlflow_cli symlink /path/to/combined_mlruns
python -m prl.scripts.mlflow_cli symlink /path/to/combined_mlruns --runs-file /other/experiment_runs.txt

# Analyze a unified or distributed run.
python -m prl.scripts.mlflow_cli analyze /path/to/run3
python -m prl.scripts.mlflow_cli analyze /path/to/run2 --distributed -f 0 -f 1 --plot --outfile summary.txt
```
