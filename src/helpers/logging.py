"""Per-module log-level filtering for loguru.

Usage in a notebook or script::

    from helpers.logging import MODULE_LEVELS, setup_logging

    # Tweak levels before setting up
    MODULE_LEVELS["scripts.generate_fold_predictions"] = "WARNING"
    MODULE_LEVELS["core.grid"] = "DEBUG"
    setup_logging()

    # Re-configure mid-session
    MODULE_LEVELS["scripts.generate_fold_predictions"] = "DEBUG"
    setup_logging()   # remove + re-add the handler
"""

import sys
from loguru import logger

# Mutable dict — edit values freely, then call setup_logging() to apply.
# Keys are Python module __name__ substrings (partial matches are fine).
# Default level for any module not listed here is DEFAULT_LEVEL.
MODULE_LEVELS: dict[str, str] = {
    # core
    "core.dataset":                        "INFO",
    "core.experiment":                     "INFO",
    "core.grid":                           "INFO",
    # preprocessing
    "preprocessing.create_rois":           "INFO",
    "preprocessing.create_datalist":       "INFO",
    "preprocessing.prepare_training_data": "INFO",
    # scripts
    "scripts.generate_fold_predictions":   "INFO",
    "scripts.inference":                   "INFO",
    "scripts.compute_performance_metrics": "INFO",
    # cli
    "cli":                                 "INFO",
}

DEFAULT_LEVEL = "INFO"

_FORMAT = "{time:HH:mm:ss} | {level:<7} | {name} - {message}"
_handler_id: int | None = None


def module_filter(record) -> bool:
    """Loguru filter that applies per-module minimum levels from MODULE_LEVELS."""
    name = record["name"]
    for module, level in MODULE_LEVELS.items():
        if module in name:
            return record["level"].no >= logger.level(level).no
    return record["level"].no >= logger.level(DEFAULT_LEVEL).no


def setup_logging(format: str = _FORMAT) -> None:
    """Install (or reinstall) the per-module filter on stderr.

    Call once at the top of a notebook, and again any time you update
    MODULE_LEVELS mid-session.
    """
    global _handler_id
    if _handler_id is not None:
        logger.remove(_handler_id)
    _handler_id = logger.add(sys.stderr, filter=module_filter, format=format)
