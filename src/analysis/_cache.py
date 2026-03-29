"""Orthogonal caching layer for run data.

Wraps load_run_data() so caching is decoupled from computation.
Cache files are pickle dicts stored under PROJECT_ROOT/analysis/.cache/.

Functions:
    cache_path: Deterministic cache file path for a run.
    cache_run: Write run data to cache.
    load_cached: Load cached data if it exists.
    cached_load_run_data: load_run_data with transparent caching.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

from loguru import logger

from helpers.paths import PROJECT_ROOT

_CACHE_DIR = PROJECT_ROOT / "analysis" / ".cache"


def cache_path(run_id: str) -> Path:
    """Deterministic cache file path for a run.

    Args:
        run_id: Experiment identifier (e.g. "roi_train2/stage1_crop_lr_sweep/run1").
            Slashes are replaced with underscores.
    """
    safe_name = run_id.replace("/", "_")
    return _CACHE_DIR / f"{safe_name}.pkl"


def cache_run(run_id: str, data: dict) -> Path:
    """Write run data to cache. Returns the cache file path."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(run_id)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.debug(f"Cached: {path.name}")
    return path


def load_cached(run_id: str) -> dict | None:
    """Load cached data if it exists. Returns None if not cached."""
    path = cache_path(run_id)
    if path.exists():
        logger.debug(f"Loading cached: {path.name}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def cached_load_run_data(
    experiment,
    dataset=None,
    use_cache: bool = True,
    **kwargs,
) -> dict | None:
    """load_run_data() with transparent caching.

    Args:
        experiment: Experiment instance, Path, or string (passed to load_run_data).
        dataset: Optional Dataset for path-based construction.
        use_cache: If True, try loading from cache first and write after computing.
        **kwargs: Forwarded to load_run_data (compute_performance, compute_mlflow).

    Returns:
        Run data dict, or None if the run directory doesn't exist.
    """
    from core.experiment import Experiment

    # Resolve run_id for cache key
    if isinstance(experiment, Experiment):
        run_id = experiment.id
    else:
        run_dir = Path(experiment)
        # Try to construct a meaningful id from the path
        run_id = str(run_dir).replace("/", "_")

    if use_cache:
        cached = load_cached(run_id)
        if cached is not None:
            return cached

    from analysis.loaders import load_run_data

    result = load_run_data(experiment, dataset=dataset, **kwargs)
    if result is None:
        return None

    #! I changed what claude did: i messed something up with imports when first adapting this,
    #!  and then couldn't figure out why the fixed version wasn't cachine
    #!  maybe it makes sense to rebuild cache every time even if use_cache is False;
    #!  or an option to rebuild the cache could be there, and then cache is 
    #!  only rebuilt if use_cash and rebuild are true
    # if use_cache:
    #     cache_run(result.get("run_name", run_id), result)
    cache_run(result.get("run_name", run_id), result)

    return result
