"""Display helpers for metric tables: formatting, renaming, column ordering.

Functions:
    format_param_value: Format a hyperparameter value for display in tables.
    extract_param: Extract a nested parameter using '#'-separated key paths.
    rename_metric: Apply display-friendly renaming to MLflow metric names.
    order_columns: Reorder DataFrame columns by semantic category.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

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
    cols_to_keep = [
        c for c in df.columns if not any(sub in c for sub in DROP_COLUMN_SUBSTRINGS)
    ]
    df = df[cols_to_keep]

    cats: dict[str, list[str]] = defaultdict(list)
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
        cat_folds = [k for k in cats[cat] if "fold" in k]
        cat_overall = set(cats[cat]) - set(cat_folds)
        ordered.extend(list(cat_overall) + sorted(cat_folds))
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]
