"""PRL experiment analysis pipeline.

Submodules:
    metrics — performance, mlflow, and display helpers
    loaders — load_run_data() for building experiment data dicts
    compile — compile_all_metrics() for cross-experiment/grid DataFrames
    _cache  — orthogonal caching layer

Typical notebook usage::

    from analysis.loaders import load_run_data
    from analysis.metrics import performance_metrics, mlflow_metrics
    from analysis.compile import compile_all_metrics

    # Single run
    data = load_run_data(experiment)
    perf = performance_metrics(data)

    # Across a grid
    df = compile_all_metrics(performance_metrics, grids=[grid])
"""
