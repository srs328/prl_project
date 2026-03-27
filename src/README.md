# Collate a distributed run into one mlruns/
python mlflow_cli.py collate /path/to/run2

# Build combined view from all runs in experiment_runs.txt
python mlflow_cli.py symlink /path/to/combined_mlruns
# or with a custom runs file:
python mlflow_cli.py symlink /path/to/combined_mlruns --runs-file /other/experiment_runs.txt

# Analyze a run (unified mlruns)
python mlflow_cli.py analyze /path/to/run3

# Analyze distributed (before collation), specific folds, with plots
python mlflow_cli.py analyze /path/to/run2 --distributed -f 0 -f 1 --plot --outfile summary.txt
