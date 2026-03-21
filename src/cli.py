"""PRL Pipeline CLI.

Unified command-line interface for the PRL detection pipeline.

Usage:
    prl preprocess roi_train2
    prl train roi_train2
    prl grid roi_train2 experiment.yaml
    prl predict /path/to/run_dir
    prl metrics /path/to/run_dir
"""

import click
from pathlib import Path


@click.group()
def cli():
    """PRL detection pipeline."""
    pass


@cli.command()
@click.argument("dataset_name")
@click.option("--expand-xy", type=int, default=None, help="X/Y expansion (overrides dataset default)")
@click.option("--expand-z", type=int, default=None, help="Z expansion (overrides dataset default)")
@click.option("--processes", type=int, default=None, help="Parallel processes (default: sequential)")
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
@click.option("--rebuild-datalist", is_flag=True, default=False, help="Rebuild datalist_template.json")
def preprocess(dataset_name, expand_xy, expand_z, processes, dry_run, rebuild_datalist):
    """Run full preprocessing pipeline for a dataset.

    DATASET_NAME is the name of the dataset (e.g., 'roi_train2').
    Looks up PROJECT_ROOT/training/{name}/dataset.yaml.
    """
    import attrs
    from core.dataset import Dataset
    from core.configs import PreprocessingConfig

    ds = Dataset(dataset_name)
    config = ds.default_preprocess

    overrides = {}
    if expand_xy is not None:
        overrides["expand_xy"] = expand_xy
    if expand_z is not None:
        overrides["expand_z"] = expand_z
    if processes is not None:
        overrides["processes"] = processes
    if dry_run:
        overrides["dry_run"] = True

    if overrides:
        config = attrs.evolve(config, **overrides)

    click.echo(f"Preprocessing {ds.name} with {config.suffix}")
    result = ds.preprocess(config, rebuild_datalist=rebuild_datalist)
    click.echo(f"Done. Datalist: {result}")


@cli.command()
@click.argument("dataset_name")
@click.option("--run-dir", type=click.Path(path_type=Path), default=None,
              help="Run directory (default: auto-increment)")
@click.option("--expand-xy", type=int, default=None)
@click.option("--expand-z", type=int, default=None)
@click.option("--epochs", type=int, default=None)
@click.option("--lr", type=float, default=None, help="Learning rate")
@click.option("--batch-size", type=int, default=None)
@click.option("--roi-size", type=int, nargs=3, default=None, help="ROI size (3 ints)")
def train(dataset_name, run_dir, expand_xy, expand_z, epochs, lr, batch_size, roi_size):
    """Train a model on a dataset.

    DATASET_NAME is the name of the dataset (e.g., 'roi_train2').
    """
    import attrs
    from core.dataset import Dataset
    from core.experiment import Experiment

    ds = Dataset(dataset_name)
    pp_config = ds.default_preprocess
    tr_config = ds.default_training

    pp_overrides = {}
    if expand_xy is not None:
        pp_overrides["expand_xy"] = expand_xy
    if expand_z is not None:
        pp_overrides["expand_z"] = expand_z
    if pp_overrides:
        pp_config = attrs.evolve(pp_config, **pp_overrides)

    tr_overrides = {}
    if epochs is not None:
        tr_overrides["num_epochs"] = epochs
    if lr is not None:
        tr_overrides["learning_rate"] = lr
    if batch_size is not None:
        tr_overrides["num_images_per_batch"] = batch_size
    if roi_size is not None:
        tr_overrides["roi_size"] = list(roi_size)
    if tr_overrides:
        tr_config = attrs.evolve(tr_config, **tr_overrides)

    exp = Experiment(ds, pp_config, tr_config, run_dir=run_dir or Experiment(ds, pp_config, tr_config, Path(".")).next_run_dir())
    click.echo(f"Training {ds.name} in {exp.run_dir}")
    exp.setup()
    exp.train()


@cli.command()
@click.argument("dataset_name")
@click.argument("experiment_config", type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Preview without writing")
@click.option("--no-prepare", is_flag=True, help="Skip data preparation")
@click.option("--hpc", is_flag=True, help="Submit to HPC instead of running locally")
@click.option("--processes", type=int, default=1, help="Parallel processes for local execution")
@click.option("--run-key", type=str, default=None, help="Launch only a specific run")
@click.option("--launch", is_flag=True, help="Generate and immediately launch")
def grid(dataset_name, experiment_config, dry_run, no_prepare, hpc, processes, run_key, launch):
    """Generate (and optionally launch) HPO experiments.

    DATASET_NAME is the dataset name. EXPERIMENT_CONFIG is a YAML file
    with param_grid and experiment_name.
    """
    from helpers.paths import load_config
    from core.dataset import Dataset
    from core.grid import ExperimentGrid

    config = load_config(experiment_config)
    ds = Dataset(dataset_name)

    eg = ExperimentGrid(
        dataset=ds,
        param_grid=config["param_grid"],
        experiment_name=config["experiment_name"],
    )

    experiments = eg.generate(dry_run=dry_run, prepare_data=not no_prepare)

    if launch and not dry_run:
        mode = "hpc" if hpc else "local"
        eg.launch(experiments=experiments, mode=mode, processes=processes,
                  dry_run=False, run_key=run_key)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--fold", type=int, default=None, help="Specific fold (default: all)")
@click.option("--dataset", "dataset_name", type=str, default=None,
              help="Dataset name (auto-detected from run_dir configs if omitted)")
def predict(run_dir, fold, dataset_name):
    """Generate fold validation predictions for a training run.

    RUN_DIR is the path to the training run directory.
    """
    from core.dataset import Dataset
    from core.experiment import Experiment

    if dataset_name is None:
        from helpers.paths import load_config
        label_config = load_config(run_dir / "label_config.json")
        # Infer dataset name from train_home
        dataset_name = Path(label_config["train_home"]).name

    ds = Dataset(dataset_name)
    exp = Experiment.from_run_dir(run_dir, ds)

    results = exp.predict(fold=fold)

    for fold_num, result in sorted(results.items()):
        status = "ok" if result == "success" else "FAILED"
        click.echo(f"  Fold {fold_num}: {status}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--test-only", is_flag=True, help="Only analyze test set")
@click.option("--output-csv", type=click.Path(path_type=Path), default=None,
              help="Save results to CSV")
@click.option("--print", "print_results", is_flag=True, help="Print detailed results")
@click.option("--dataset", "dataset_name", type=str, default=None,
              help="Dataset name (auto-detected if omitted)")
def metrics(run_dir, test_only, output_csv, print_results, dataset_name):
    """Compute performance metrics for a training run.

    RUN_DIR is the path to the training run directory.
    """
    from core.dataset import Dataset
    from core.experiment import Experiment

    if dataset_name is None:
        from helpers.paths import load_config
        label_config = load_config(run_dir / "label_config.json")
        dataset_name = Path(label_config["train_home"]).name

    ds = Dataset(dataset_name)
    exp = Experiment.from_run_dir(run_dir, ds)

    if output_csv is None:
        output_csv = run_dir / "performance_metrics.csv"

    df = exp.evaluate(test_only=test_only, output_csv=output_csv,
                      print_results=print_results)

    if df is not None:
        click.echo(f"Metrics computed for {len(df)} cases -> {output_csv}")
    else:
        click.echo("No results to report.")


if __name__ == "__main__":
    cli()
