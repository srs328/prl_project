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
@click.option(
    "--expand-xy",
    type=int,
    default=None,
    help="X/Y expansion (overrides dataset default)",
)
@click.option(
    "--expand-z", type=int, default=None, help="Z expansion (overrides dataset default)"
)
@click.option(
    "--images",
    type=str,
    multiple=True,
    default=None,
    help="Image channels (e.g. --images flair --images phase)",
)
@click.option(
    "--processes",
    type=int,
    default=None,
    help="Parallel processes (default: sequential)",
)
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
@click.option(
    "--rebuild-datalist",
    is_flag=True,
    default=False,
    help="Rebuild datalist_template.json",
)
def preprocess(
    dataset_name, expand_xy, expand_z, images, processes, dry_run, rebuild_datalist
):
    """Run full preprocessing pipeline for a dataset.

    DATASET_NAME is the name of the dataset (e.g., 'roi_train2').
    Looks up PROJECT_ROOT/training/{name}/dataset.yaml.
    """
    import attrs
    from core.dataset import Dataset
    from core.experiment import Experiment

    ds = Dataset(dataset_name)
    config = ds.default_preprocess

    overrides = {}
    if expand_xy is not None:
        overrides["expand_xy"] = expand_xy
    if expand_z is not None:
        overrides["expand_z"] = expand_z
    if images:
        overrides["images"] = images
    if processes is not None:
        overrides["processes"] = processes
    if dry_run:
        overrides["dry_run"] = True

    if overrides:
        config = attrs.evolve(config, **overrides)

    # Create datalist template (Dataset responsibility — fold assignments)
    ds.create_datalist(rebuild=rebuild_datalist)

    # Create ROIs and prepare data (Experiment responsibility)
    exp = Experiment(ds, config, ds.default_training, run_dir=Path("."))
    click.echo(f"Preprocessing {ds.name} with {config.datalist_suffix}")
    exp.create_rois()
    result = exp.prepare_data()
    click.echo(f"Done. Datalist: {result}")


@cli.command()
@click.argument("dataset_name")
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Run directory (default: auto-increment)",
)
@click.option("--expand-xy", type=int, default=None)
@click.option("--expand-z", type=int, default=None)
@click.option("--images", type=str, multiple=True, default=None, help="Image channels")
@click.option("--epochs", type=int, default=None)
@click.option("--lr", type=float, default=None, help="Learning rate")
@click.option("--batch-size", type=int, default=None)
@click.option("--num-crops-per-image", type=int, default=None)
@click.option("--roi-size", type=int, nargs=3, default=None, help="ROI size (3 ints)")
@click.option("--algos", type=str, multiple=True, default=None, help="Models to use")
@click.option("--init-only", is_flag=True, help="Just create the run dir")
def train(
    dataset_name,
    run_dir,
    expand_xy,
    expand_z,
    images,
    epochs,
    lr,
    batch_size,
    num_crops_per_image,
    roi_size,
    algos,
    init_only,
):
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
    if images:
        pp_overrides["images"] = images
    if pp_overrides:
        pp_config = attrs.evolve(pp_config, **pp_overrides)

    tr_overrides = {}
    if epochs is not None:
        tr_overrides["num_epochs"] = epochs
    if lr is not None:
        tr_overrides["learning_rate"] = lr
    if batch_size is not None:
        tr_overrides["num_images_per_batch"] = batch_size
    if num_crops_per_image is not None:
        tr_overrides["num_crops_per_image"] = num_crops_per_image
    if roi_size is not None:
        tr_overrides["roi_size"] = list(roi_size)
    if algos:
        tr_overrides["algos"] = algos
    if tr_overrides:
        tr_config = attrs.evolve(tr_config, **tr_overrides)

    exp = Experiment(
        ds,
        pp_config,
        tr_config,
        run_dir=run_dir
        or Experiment(ds, pp_config, tr_config, Path(".")).next_run_dir(),
    )
    exp.setup()
    if init_only:
        click.echo(f"Initializing {exp.run_dir}")
        return
    click.echo(f"Training {ds.name} in {exp.run_dir}")
    exp.train()


@cli.command()
@click.argument("experiment_config", type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Preview without writing")
@click.option("--no-prepare", is_flag=True, help="Skip data preparation")
@click.option("--hpc", is_flag=True, help="Submit to HPC instead of running locally")
@click.option(
    "--processes", type=int, default=1, help="Parallel processes for local execution"
)
@click.option("--run-key", type=str, default=None, help="Launch only a specific run")
@click.option("--launch", is_flag=True, help="Generate and immediately launch")
@click.option(
    "--overwrite", is_flag=True, help="To overwrite existing experiment run folders"
)
@click.option(
    "--validate", is_flag=True, help="To validate that all datalist files exist"
)
def grid(
    experiment_config,
    dry_run,
    no_prepare,
    hpc,
    processes,
    run_key,
    launch,
    overwrite,
    validate,
):
    """Generate (and optionally launch) HPO experiments.

    EXPERIMENT_CONFIG is a YAML/JSON file with dataset_name, experiment_name,
    and param_grid.
    """
    from core.grid import ExperimentGrid

    eg = ExperimentGrid.from_config(experiment_config)

    experiments = eg.generate(
        dry_run=dry_run,
        prepare_data=not no_prepare,
        validate=validate,
        overwrite=overwrite,
    )

    if launch:
        mode = "hpc" if hpc else "local"
        eg.launch(
            experiments=experiments,
            mode=mode,
            processes=processes,
            dry_run=dry_run,
            run_key=run_key,
        )


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--fold", type=int, default=None, help="Specific fold (default: all)")
@click.option(
    "--dataset",
    "dataset_name",
    type=str,
    default=None,
    help="Dataset name (auto-detected from run_dir configs if omitted)",
)
def predict(run_dir, fold, dataset_name):
    """Generate fold validation predictions for a training run.

    RUN_DIR is the path to the training run directory.
    """
    from core.dataset import Dataset
    from core.experiment import Experiment

    if dataset_name is None:
        from helpers.paths import load_config

        label_config = load_config(run_dir / "label_config.json")
        dataset_name = label_config["dataset_name"]

    ds = Dataset(dataset_name)
    exp = Experiment.from_run_dir(run_dir, ds)

    results = exp.predict(fold=fold)

    for fold_num, result in sorted(results.items()):
        status = "ok" if result == "success" else "FAILED"
        click.echo(f"  Fold {fold_num}: {status}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--test-only", is_flag=True, help="Only analyze test set")
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Save results to CSV",
)
@click.option("--print", "print_results", is_flag=True, help="Print detailed results")
# @click.option("--print-file", "print_results", type=str, help="File to print results to") #?can't have the two share?
@click.option("--print-file", "print_file", type=str, help="File to print results to")
@click.option(
    "--dataset",
    "dataset_name",
    type=str,
    default=None,
    help="Dataset name (auto-detected if omitted)",
)
def metrics(run_dir, test_only, output_csv, print_results, print_file, dataset_name):
    """Compute performance metrics for a training run.

    RUN_DIR is the path to the training run directory.
    """
    from core.dataset import Dataset
    from core.experiment import Experiment

    if dataset_name is None:
        from helpers.paths import load_config

        label_config = load_config(run_dir / "label_config.json")
        dataset_name = label_config["dataset_name"]

    ds = Dataset(dataset_name)
    exp = Experiment.from_run_dir(run_dir, ds)

    if output_csv is not None and not output_csv.is_absolute():
        output_csv = run_dir / output_csv

    if print_file:
        print_results = print_file
    df = exp.evaluate(
        test_only=test_only, output_csv=output_csv, print_results=print_results
    )

    if df is not None:
        click.echo(f"Metrics computed for {len(df)} cases -> {output_csv}")
    else:
        click.echo("No results to report.")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("subject", required=False, default=None)
@click.option(
    "--data-root",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Data root for resolving subject names (default: PRL_DATA_ROOT)",
)
@click.option(
    "--all",
    "process_all",
    is_flag=True,
    help="Process every subject under --data-root (requires explicit --data-root)",
)
@click.option(
    "--subjects-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Text file with subject names (one per line), resolved against --data-root",
)
@click.option(
    "--processes",
    type=int,
    default=None,
    help="Parallel processes for multi-subject inference (default: sequential)",
)
def infer(run_dir, subject, data_root, process_all, subjects_file, processes):
    """Run trained model on fresh subject(s).

    RUN_DIR is the trained model directory.
    SUBJECT is a subject folder name (resolved under --data-root) or absolute path.
    Use --all to process every subject under --data-root, or --subjects-file for a list.
    """
    import re
    from scripts.inference import infer_subject
    from helpers.paths import DATA_ROOT

    if data_root is None:
        data_root = DATA_ROOT

    # Resolve subject directories
    subject_dirs = []

    if process_all:
        if not subject and not subjects_file:
            # Scan data_root for subject directories
            subject_dirs = sorted(
                p
                for p in data_root.iterdir()
                if p.is_dir() and re.match(r"^sub\d+", p.name)
            )
            if not subject_dirs:
                click.echo(f"No subject directories found under {data_root}")
                return
        else:
            raise click.UsageError(
                "--all cannot be combined with SUBJECT or --subjects-file"
            )

    elif subjects_file is not None:
        if subject:
            raise click.UsageError("Cannot combine SUBJECT with --subjects-file")
        with open(subjects_file) as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                p = Path(name)
                if p.is_absolute():
                    subject_dirs.append(p)
                else:
                    subject_dirs.append(data_root / name)

    elif subject is not None:
        p = Path(subject)
        if p.is_absolute() and p.exists():
            subject_dirs.append(p)
        else:
            subject_dirs.append(data_root / subject)

    else:
        raise click.UsageError("Provide SUBJECT, --all, or --subjects-file")

    # Validate all dirs exist
    for sd in subject_dirs:
        if not sd.exists():
            raise click.UsageError(f"Subject directory not found: {sd}")

    click.echo(f"Processing {len(subject_dirs)} subject(s) using {run_dir}")

    tasks = [
        {"run_dir": run_dir, "subject_dir": sd, "data_root": data_root}
        for sd in subject_dirs
    ]

    if processes is not None and len(tasks) > 1:
        from helpers.parallel import BetterPool
        from tqdm import tqdm

        click.echo(f"Using {processes} parallel processes")
        with BetterPool(processes) as pool:
            results = pool.imap_unordered(_infer_wrapper, tasks)
            for name, result in tqdm(results, total=len(tasks)):
                click.echo(f"  {name}: {result}")
    else:
        for task in tasks:
            sd = task["subject_dir"]
            click.echo(f"\n--- {sd.name} ---")
            result = infer_subject(**task)
            click.echo(f"Output: {result}")

    click.echo(f"\nDone. Processed {len(subject_dirs)} subject(s).")


def _infer_wrapper(kwargs):
    """Wrapper for BetterPool — unpacks dict args and returns (name, result)."""
    from scripts.inference import infer_subject

    name = kwargs["subject_dir"].name
    result = infer_subject(**kwargs)
    return name, result


if __name__ == "__main__":
    cli()

# prl infer --data-root $inf_dataroot --subjects-file $subjects_list  $run_dir
