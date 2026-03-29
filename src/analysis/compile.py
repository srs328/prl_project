"""Compile training metrics across experiments and grids into unified DataFrames.

Functions:
    compile_experiment_metrics: Compile metrics for a single experiment.
    compile_grid_metrics: Compile metrics for all runs in an ExperimentGrid.
    compile_all_metrics: Compile across multiple experiments and grids.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from core.experiment import Experiment
from core.grid import ExperimentGrid
from analysis._cache import cached_load_run_data
from analysis.metrics.display import format_param_value, extract_param

if TYPE_CHECKING:
    from typing import Callable


def _build_generic_row(
    experiment_id: str,
    cached: dict,
    params_to_gather: list[str] | None,
) -> dict:
    """Build a single row dict with experiment ID and requested hyper params."""
    row = {"ID": experiment_id}
    hyper_params = cached.get("hyper_params", {})

    if params_to_gather:
        for key in params_to_gather:
            val = extract_param(hyper_params, key)
            display_key = key.split("#")[-1]
            row[display_key] = format_param_value(key, val)

    return row


def compile_experiment_metrics(
    experiment: Experiment | Path | str,
    func: Callable,
    params_to_gather: list[str] | None = None,
    use_cache: bool = True,
) -> dict:
    """Compile metrics for a single experiment using a metric function.

    Args:
        experiment: Experiment instance, run_dir Path, or string path.
        func: Metric function with signature (experiment_data: dict) -> dict.
        params_to_gather: Hyper param keys to extract as columns (e.g. "loss#weight").
        use_cache: Whether to use/write disk cache.

    Returns:
        Flat dict suitable for one DataFrame row.
    """
    if not isinstance(experiment, Experiment):
        experiment = Experiment.from_run_dir(experiment)

    logger.info(f"Starting Experiment {experiment.id}")
    cached = cached_load_run_data(experiment, use_cache=use_cache)
    if cached is None:
        return {"ID": experiment.id, "status": "missing"}

    row = _build_generic_row(experiment.id, cached, params_to_gather)
    return {**row, **func(cached)}


def compile_grid_metrics(
    grid: ExperimentGrid | Path | str,
    func: Callable,
    params_to_gather: list[str] | None = None,
    use_cache: bool = True,
    runs_to_skip: dict | None = None,
) -> list[dict]:
    """Compile metrics for all runs in a grid stage.

    Args:
        grid: ExperimentGrid instance or path to grid home directory.
        func: Metric function with signature (experiment_data: dict) -> dict.
        params_to_gather: Param keys to extract as columns. If None,
            auto-detects from the grid's param_grid keys.
        use_cache: Whether to use/write disk cache.
        runs_to_skip: Dict of run names to skip.

    Returns:
        List of row dicts (one per run).
    """
    if not isinstance(grid, ExperimentGrid):
        grid = ExperimentGrid.from_home_dir(grid)
    logger.info(f"Starting grid {grid.experiment_name}")

    if params_to_gather is None:
        params_to_gather = []
        for section in ("training", "preprocessing"):
            for key in grid.param_grid.get(section, {}):
                params_to_gather.append(key)

    rows = []
    for experiment in grid.experiments:
        run_dir = experiment.run_dir
        if not (run_dir / "ensemble_output").exists() or (
            runs_to_skip is not None and run_dir.name in runs_to_skip
        ):
            logger.debug(f"Skipped {experiment.id} for {grid.experiment_name}")
            continue
        row = compile_experiment_metrics(
            experiment, func, params_to_gather, use_cache=use_cache
        )
        rows.append(row)

    logger.info(f"Compiled [{', '.join([r['ID'] for r in rows])}]")
    return rows


def compile_all_metrics(
    func: Callable,
    experiments: list[Experiment | Path | str] | None = None,
    grids: list[ExperimentGrid | Path | str] | None = None,
    params_to_gather: list[str] | None = None,
    use_cache: bool = True,
    runs_to_skip: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compile metrics across multiple experiments and grids.

    Args:
        func: Metric function with signature (experiment_data: dict) -> dict.
        experiments: List of individual experiments to compile.
        grids: List of experiment grids to compile.
        params_to_gather: Param keys to extract as columns.
        use_cache: Whether to use/write disk cache.
        runs_to_skip: Dict of run names to skip.

    Returns:
        DataFrame with one row per run.
    """
    if runs_to_skip is None:
        runs_to_skip = {}

    data = []
    if experiments is not None:
        get_metrics = partial(
            compile_experiment_metrics,
            func=func,
            params_to_gather=params_to_gather,
            use_cache=use_cache,
            **kwargs,
        )
        data.extend([get_metrics(experiment) for experiment in experiments])

    if grids is not None:
        get_metrics = partial(
            compile_grid_metrics,
            func=func,
            params_to_gather=params_to_gather,
            use_cache=use_cache,
            runs_to_skip=runs_to_skip,
            **kwargs,
        )
        for grid in grids:
            data.extend(get_metrics(grid))

    result = pd.DataFrame(data)
    param_cols = []
    if params_to_gather:
        param_cols = [k.split("#")[-1] for k in params_to_gather]
    for col in param_cols:
        if col in result.columns:
            result[col] = result[col].fillna("default")

    return result
