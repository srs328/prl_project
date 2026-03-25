"""Compile training metrics across grid search stages into unified DataFrames.

Usage:
    # As library
    from scripts.compile_run_metrics import compile_all_metrics
    df = compile_all_metrics("roi_train2", ["stage3_numcrops_bkd_constwt115"])

    # Via CLI
    prl compile roi_train2 --stage stage3_numcrops_bkd_constwt115
"""

from __future__ import annotations

import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import attrs
import numpy as np
import pandas as pd
from loguru import logger

from helpers.paths import PROJECT_ROOT, TRAIN_ROOT

EXPERIMENT_KEYS = {
    "stage1": "stage1_crop_lr_sweep",
    "stage2": "stage2_numcrops_dicece",
    "stage3": "stage3_numcrops_bkd_constwt115",
    "stage4": "stage4_sweep_dicece_wts",
    "stage5": "stage5_sweep_dicecewt_nbatch",
    "lambda_test": "test_dicece_lambda",
}

# ---------------------------------------------------------------------------
# Parameter display
# ---------------------------------------------------------------------------

_TORCH_TENSOR_RE = re.compile(r"\$torch\.tensor\(\[(.+?)\]\)")

METRIC_DISPLAY_NAMES = {
    "val_class/acc_0": "lesion/acc",
    "val_class/acc_1": "rim/acc",
}

COLUMN_ORDER_CATEGORIES = ["run_info", "rim", "lesion", "loss", "val"]

DROP_COLUMN_SUBSTRINGS = {"acc_min", "loss_max"}


def format_param_value(key: str, value) -> str:
    """Format a hyperparameter value for display in tables.

    Handles torch tensor strings, None, lists, bools, dicts (loss), scalars.
    """
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        m = _TORCH_TENSOR_RE.search(value)
        if m:
            return m.group(1).replace(" ", "")
        return value
    if isinstance(value, dict):
        # Loss dict: summarize key fields
        parts = []
        if "weight" in value:
            parts.append(f"wt={format_param_value(key, value['weight'])}")
        if "include_background" in value:
            parts.append("bkd" if value["include_background"] else "nobkd")
        if "lambda_dice" in value:
            parts.append(f"ld={value['lambda_dice']}")
        if "lambda_ce" in value:
            parts.append(f"lce={value['lambda_ce']}")
        if parts:
            return ";".join(parts)
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def extract_param(params: dict, key_path: str):
    """Extract a nested parameter using '#'-separated key paths.

    e.g., extract_param(params, "loss#weight") walks params["loss"]["weight"].
    Returns None if any key is missing.
    """
    obj = params
    for k in key_path.split("#"):
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def rename_metric(metric_name: str) -> str:
    """Apply display-friendly renaming to MLflow metric names."""
    for pattern, replacement in METRIC_DISPLAY_NAMES.items():
        if pattern in metric_name:
            return metric_name.replace(pattern, replacement)
    return metric_name


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns by semantic category and drop noisy metrics."""
    # Drop noisy columns
    cols_to_keep = [
        c for c in df.columns
        if not any(sub in c for sub in DROP_COLUMN_SUBSTRINGS)
    ]
    df = df[cols_to_keep]

    cats = defaultdict(list)
    for col in df.columns:
        if col in ("stage", "run", "status") or not any(
            cat in col for cat in ["rim", "lesion", "train", "val"]
        ):
            cats["run_info"].append(col)
        elif "rim" in col:
            cats["rim"].append(col)
        elif "lesion" in col:
            cats["lesion"].append(col)
        elif "train" in col:
            cats["loss"].append(col)
        elif "val" in col:
            cats["val"].append(col)

    ordered = []
    for cat in COLUMN_ORDER_CATEGORIES:
        ordered.extend(sorted(cats[cat]))
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_CACHE_DIR = PROJECT_ROOT / "analysis" / ".cache"


def _cache_path(stage_name: str, run_name: str) -> Path:
    return _CACHE_DIR / f"{stage_name}_{run_name}.pkl"


def load_or_cache_run(
    dataset_name: str,
    stage_name: str,
    run_name: str,
    run_dir: Path,
    use_cache: bool = True,
) -> dict | None:
    """Load cached run data, or compute and cache it.

    Returns a dict with keys: run_name, run_dir, cases, mlflow_aggregated,
    hyper_params, fold_data. Returns None if the run directory doesn't exist.
    """
    if not run_dir.exists():
        logger.warning(f"Run dir does not exist: {run_dir}")
        return None

    cache_file = _cache_path(stage_name, run_name)

    if use_cache and cache_file.exists():
        logger.debug(f"Loading cached: {cache_file.name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Build from scratch
    from core.dataset import Dataset
    from core.experiment import Experiment
    from scripts.analyze_mlflow_runs import (
        analyze_unified_mlruns,
        aggregate_metrics,
    )

    ds = Dataset(dataset_name)
    try:
        exp = Experiment.from_run_dir(run_dir, ds)
    except Exception as e:
        logger.warning(f"Could not load {run_dir}: {e}")
        return None

    # hyper_params (actual runtime values)
    hp = exp.hyper_params
    hyper_params = attrs.asdict(hp) if hp is not None else {}

    # MLflow metrics
    try:
        fold_data = analyze_unified_mlruns(run_dir)
        mlflow_aggregated = dict(aggregate_metrics(fold_data)) if fold_data else {}
    except Exception as e:
        logger.warning(f"MLflow read failed for {run_dir}: {e}")
        fold_data = {}
        mlflow_aggregated = {}

    # Cases (the expensive part over SMB)
    try:
        cases = exp.cases or []
    except Exception as e:
        logger.warning(f"Cases build failed for {run_dir}: {e}")
        cases = []

    # Convert fold_data to plain dicts for pickling
    fold_data_plain = {}
    for k, v in fold_data.items():
        if isinstance(v, dict):
            fold_data_plain[k] = dict(v)
        else:
            fold_data_plain[k] = v

    result = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "cases": cases,
        "mlflow_aggregated": mlflow_aggregated,
        "mlflow_fold_data": fold_data_plain,
        "hyper_params": hyper_params,
        "timestamp": time.time(),
    }

    # Write cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    logger.debug(f"Cached: {cache_file.name}")

    return result


# ---------------------------------------------------------------------------
# Metric compilation
# ---------------------------------------------------------------------------


def _build_row(
    stage_name: str,
    run_name: str,
    cached: dict,
    params_to_gather: list[str] | None,
) -> dict:
    """Build a single row dict from cached run data."""
    row = {"stage": stage_name, "run": run_name}

    hyper_params = cached["hyper_params"]
    aggregated = cached["mlflow_aggregated"]

    # Status
    if not aggregated:
        row["status"] = "incomplete"
    else:
        row["status"] = "complete"

    # Extract requested params from hyper_params
    if params_to_gather:
        for key in params_to_gather:
            val = extract_param(hyper_params, key)
            display_key = key.split("#")[-1]  # loss#weight -> weight
            row[display_key] = format_param_value(key, val)

    # Flatten MLflow aggregated metrics
    by_prefix = defaultdict(dict)
    for metric_name in aggregated:
        renamed = rename_metric(metric_name)
        if "/" in renamed:
            prefix = renamed.split("/")[0]
            by_prefix[prefix][renamed] = aggregated[metric_name]
        else:
            by_prefix["other"][renamed] = aggregated[metric_name]

    fold_nums = sorted(
        k for k in (cached.get("mlflow_fold_data") or {}).keys()
        if isinstance(k, int)
    )

    for prefix in sorted(by_prefix.keys()):
        for metric_name in sorted(by_prefix[prefix].keys()):
            metric_data = aggregated.get(
                # reverse-lookup original name for aggregated access
                next(
                    (orig for orig, data in aggregated.items()
                     if rename_metric(orig) == metric_name),
                    metric_name,
                ),
                {},
            )

            if "stats" not in metric_data:
                continue

            stats = metric_data["stats"]
            for stat in ["mean", "std", "min", "max"]:
                row[f"{metric_name}_{stat}"] = np.round(stats[stat], 4)

            # Per-fold final/max for val_class metrics
            if "val_class" not in metric_name and ("rim/acc" not in metric_name and "lesion/acc" not in metric_name):
                continue
            for fold_num in fold_nums:
                if fold_num not in metric_data:
                    continue
                values = metric_data[fold_num]
                row[f"fold{fold_num}-{metric_name}_final"] = np.round(
                    values[-1], 4
                )
                row[f"fold{fold_num}-{metric_name}_max"] = np.round(
                    np.max(values), 4
                )

    return row


def compile_stage_metrics(
    dataset_name: str,
    stage_name: str,
    params_to_gather: list[str] | None = None,
    use_cache: bool = True,
    skip_runs: dict | None = None
) -> pd.DataFrame:
    """Compile MLflow training metrics for all runs in a grid stage.

    Args:
        dataset_name: e.g. "roi_train2"
        stage_name: e.g. "stage3_numcrops_bkd_constwt115"
        params_to_gather: Param keys to extract as columns (e.g. "loss#weight").
            If None, auto-detects from the grid's param_grid keys.
        use_cache: Whether to use/write disk cache.

    Returns:
        DataFrame with one row per run.
    """
    from core.grid import ExperimentGrid

    work_home = TRAIN_ROOT / dataset_name
    config_path = work_home / stage_name / "experiment_config.yaml"
    if not config_path.exists():
        config_path = config_path.with_suffix(".json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"No experiment config found at {work_home / stage_name}"
        )

    grid = ExperimentGrid.from_config(config_path)
    run_params_list = grid.run_params(*grid.get_info())

    # Auto-detect params if not specified
    if params_to_gather is None:
        params_to_gather = []
        for section in ("training", "preprocessing"):
            for key in grid.param_grid.get(section, {}):
                params_to_gather.append(key)

    rows = []
    for run_info in run_params_list:
        rn = run_info["run_name"]
        run_dir = work_home / stage_name / rn
        #FIXME i didn't actually submit all the runs originally created and want to skip those here.
        #   For now will use presence of "ensemble_output" to decide whether the run was completed
        if not (run_dir / "ensemble_output").exists() or (skip_runs is not None and rn in skip_runs):
            logger.debug(f"Skipped {rn} for {stage_name}")
            continue
        qualified_run_name = f"{stage_name}/{rn}"

        cached = load_or_cache_run(
            dataset_name, stage_name, rn, run_dir, use_cache
        )
        if cached is None:
            rows.append({
                "stage": stage_name, "run": qualified_run_name, "status": "missing"
            })
            continue

        row = _build_row(stage_name, qualified_run_name, cached, params_to_gather)
        rows.append(row)
    logger.info(f"Compiled [{', '.join([r['run'] for r in rows])}]")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = order_columns(df)
    return df


def compile_standalone_run(
    dataset_name: str,
    label: str,
    run_dir: Path,
    params_to_gather: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Compile metrics for a single standalone run (not part of a grid).

    Args:
        dataset_name: e.g. "roi_train2"
        label: Display label for this run (used as stage name).
        run_dir: Path to the run directory (absolute or relative to TRAIN_ROOT/dataset).
        params_to_gather: Param keys to extract.
        use_cache: Whether to use disk cache.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = TRAIN_ROOT / dataset_name / run_dir

    run_name = run_dir.name
    cached = load_or_cache_run(
        dataset_name, label, run_name, run_dir, use_cache
    )
    if cached is None:
        return pd.DataFrame([{"stage": label, "run": run_name, "status": "missing"}])

    row = _build_row(label, run_name, cached, params_to_gather)
    return pd.DataFrame([row])


def discover_stages(dataset_name: str) -> list[str]:
    """Find all grid experiment directories under TRAIN_ROOT/{dataset_name}."""
    work_home = TRAIN_ROOT / dataset_name
    stages = []
    for d in sorted(work_home.iterdir()):
        if not d.is_dir():
            continue
        if (d / "experiment_config.yaml").exists() or (d / "experiment_config.json").exists():
            stages.append(d.name)
    return stages


def compile_all_metrics(
    dataset_name: str,
    stages: list[str] | None = None,
    params_to_gather: list[str] | None = None,
    extra_runs: list[tuple[str, str | Path]] | None = None,
    use_cache: bool = True,
    skip_runs: dict | None = None
) -> pd.DataFrame:
    """Compile metrics across multiple grid stages into one DataFrame.

    Args:
        dataset_name: e.g. "roi_train2"
        stages: Stage names. If None, auto-discovers all stages.
        params_to_gather: Union of all param keys to show.
        extra_runs: Optional list of (label, run_dir) for standalone runs.
        use_cache: Whether to use disk cache.

    Returns:
        DataFrame with 'stage' column identifying source.
    """
    if stages is None:
        stages = discover_stages(dataset_name)
        logger.info(f"Discovered stages: {stages}")
    if skip_runs is None:
        skip_runs = {}

    dfs = []
    for stage in stages:
        logger.info(f"Compiling {stage}...")
        try:
            df = compile_stage_metrics(
                dataset_name, stage, params_to_gather, use_cache, skip_runs.get(stage)
            )
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to compile {stage}: {e}")

    if extra_runs:
        for label, run_dir in extra_runs:
            logger.info(f"Compiling standalone: {label}")
            try:
                df = compile_standalone_run(
                    dataset_name, label, run_dir, params_to_gather, use_cache
                )
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to compile {label}: {e}")

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    # Fill missing param columns with "default" for runs that didn't vary them
    param_cols = []
    if params_to_gather:
        param_cols = [k.split("#")[-1] for k in params_to_gather]
    for col in param_cols:
        if col in result.columns:
            result[col] = result[col].fillna("default")

    return result
