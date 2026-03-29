"""Data loading for experiment analysis.

Replaces the monolithic load_or_cache_run() from compile_run_metrics.py.
Each data section (cases, performance, mlflow) is independently toggleable.

Functions:
    load_run_data: Load all raw data for a run into a dict.
"""

from __future__ import annotations

import time
from pathlib import Path

import attrs
from loguru import logger

from core.experiment import Experiment
from core.dataset import Dataset


def load_run_data(
    experiment: Experiment | Path | str,
    dataset: Dataset | None = None,
    compute_performance: bool = True,
    compute_mlflow: bool = True,
) -> dict | None:
    """Load all raw data for a single run.

    Each section is independently toggleable so callers can skip expensive
    work (e.g. NIfTI loading for performance, or SMB reads for MLflow).

    Args:
        experiment: An Experiment instance, a run_dir Path, or a string path.
        dataset: Optional Dataset to pass when constructing from path.
        compute_performance: Load NIfTIs and compute casewise stats.
        compute_mlflow: Parse MLflow metric files and aggregate.

    Returns:
        Dict with keys: run_name, run_dir, cases, case_performance,
        mlflow_aggregated, mlflow_fold_data, hyper_params, timestamp.
        None if the run directory doesn't exist.
    """
    if not isinstance(experiment, Experiment):
        run_dir = Path(experiment)
        if not run_dir.exists():
            logger.warning(f"Run dir does not exist: {run_dir}")
            return None
        experiment = Experiment.from_run_dir(run_dir, dataset)

    run_dir = experiment.run_dir

    # Hyper params (actual runtime values)
    hp = experiment.hyper_params
    hyper_params = attrs.asdict(hp) if hp is not None else {}

    # Cases (DataFrame → list of dicts for serialization)
    t0 = time.perf_counter()
    try:
        cases_df = experiment.cases
        cases = []
        for idx, row in cases_df.iterrows():
            subid, lesion_index = idx
            cases.append({
                "subid": subid,
                "lesion_index": lesion_index,
                "split": row["split"],
                "case_type": row["case_type"],
                "image": str(row["image"]),
                "label": str(row["label"]),
                "inference": str(row["inference"]) if row["inference"] is not None else None,
            })
    except Exception as e:
        logger.warning(f"Cases build failed for {run_dir}: {e}")
        cases = []
    logger.debug(f"Building cases took ({time.perf_counter() - t0:.2f}s)")

    # Performance metrics (NIfTI loading — expensive)
    case_performance = []
    if compute_performance:
        from analysis.metrics.performance import compute_casewise_stats

        t0 = time.perf_counter()
        case_performance = compute_casewise_stats(cases)
        logger.debug(f"Casewise stats took ({time.perf_counter() - t0:.2f}s)")

    # MLflow metrics
    mlflow_fold_data = {}
    mlflow_aggregated = {}
    if compute_mlflow:
        from analysis.metrics.mlflow import analyze_unified_mlruns, aggregate_metrics

        t0 = time.perf_counter()
        try:
            fold_data = analyze_unified_mlruns(run_dir)
            mlflow_aggregated = aggregate_metrics(fold_data) if fold_data else {}
            # Convert fold_data to plain dicts for pickling
            mlflow_fold_data = {
                k: dict(v) if isinstance(v, dict) else v
                for k, v in fold_data.items()
            }
        except Exception as e:
            logger.warning(f"MLflow read failed for {run_dir}: {e}")
        logger.debug(f"MLflow loading took ({time.perf_counter() - t0:.2f}s)")

    return {
        "run_name": experiment.id,
        "run_dir": str(run_dir),
        "cases": cases,
        "case_performance": case_performance,
        "mlflow_aggregated": mlflow_aggregated,
        "mlflow_fold_data": mlflow_fold_data,
        "hyper_params": hyper_params,
        "timestamp": time.time(),
    }
