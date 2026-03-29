"""Metric functions for PRL experiment analysis.

Submodules:
    performance — confusion matrix, derived metrics, casewise stats, performance_metrics()
    mlflow — MLflow metric loading, aggregation, mlflow_metrics()
    display — formatting, renaming, column ordering helpers
"""

from analysis.metrics.performance import (
    get_confusion_matrix,
    compute_derived_metrics,
    compute_casewise_stats,
    performance_metrics,
)
from analysis.metrics.mlflow import (
    load_metrics_from_file,
    analyze_unified_mlruns,
    analyze_distributed_mlruns,
    aggregate_metrics,
    mlflow_metrics,
)
from analysis.metrics.display import (
    format_param_value,
    extract_param,
    rename_metric,
    order_columns,
)
