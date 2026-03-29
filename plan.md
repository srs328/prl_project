# Refactoring Plan: Analysis Pipeline, Core Modules, and Training Configs

## Overview

Four interconnected areas, ordered by dependency:
1. **Phase 1 — Core module improvements** (foundation everything else depends on)
2. **Phase 2 — Analysis pipeline redesign** (`src/analysis/`)
3. **Phase 3 — Image analysis module** (`src/analysis/image/`)
4. **Phase 4 — Cleanup and integration**

Phases 2 and 3 are independent of each other (both depend only on Phase 1).

---

## Phase 1: Core Module Improvements

### 1A. Dataset gets a `preprocess` parameter and `cases` DataFrame

**Problem**: Dataset holds subject lists and fold assignments but doesn't resolve full file paths. You need to create an Experiment (which requires an AlgoConfig) just to get usable paths — even when you only want to look at images.

**Key insight**: Path resolution depends on `PreprocessingConfig` (images, expand_xy, expand_z determine filenames), NOT on AlgoConfig. So Dataset can own path resolution if it has a preprocessing config.

**Changes to `src/core/dataset.py`:**

```python
class Dataset:
    def __init__(self, name: str, preprocess: PreprocessingConfig | None = None):
        # ... existing init ...
        self._preprocess = preprocess  # Optional override

    @property
    def preprocess(self) -> PreprocessingConfig:
        """Active preprocessing config (explicit or dataset default)."""
        return self._preprocess or self.default_preprocess

    @cached_property
    def cases(self) -> pd.DataFrame:
        """DataFrame of all cases with resolved absolute paths.
        Columns: subid, lesion_index, split, case_type, image, label, subject_dir
        Index: (subid, lesion_index)
        """
        return self._build_cases_df()
```

**Notebook usage becomes:**
```python
# Before (needs Experiment just for paths):
ds = Dataset("roi_train2")
exp = Experiment.from_run_dir(run_dir, ds)
cases = pd.DataFrame(exp.cases).set_index(["subid", "lesion_index"])

# After:
ds = Dataset("roi_train2")
ds.cases  # DataFrame with resolved paths, no Experiment needed

# Non-default preprocessing:
ds = Dataset("roi_train2", preprocess=PreprocessingConfig(expand_xy=30))
```

**Methods that become unnecessary on Dataset** (can be removed):
- `subject_dir()` — column in `cases` DataFrame
- `lesion_dir()` — derivable from cases rows
- `get_images()` — resolved in `cases` DataFrame
- `parse_stacked_image_name()` — rarely used utility

**New `Subject` class** (flesh out the existing stub):
```python
class Subject:
    """Lightweight handle for a single subject's data."""
    def __init__(self, subid: int, dataset: Dataset):
        self.subid = subid
        self.dataset = dataset
        self.dir = dataset.data_root / dataset.subject_session(subid)

    @cached_property
    def bounding_boxes(self) -> list[tuple[int, list[int]]]:
        bbox_file = self.dir / f"lstai_bounding_boxes_{self.dataset.preprocess.suffix}.txt"
        return _parse_bounding_boxes(bbox_file)

    def cases(self) -> pd.DataFrame:
        """All cases for this subject from the dataset."""
        return self.dataset.cases.loc[self.subid]

    def load_nifti(self, name: str) -> nib.Nifti1Image:
        """Load a top-level NIfTI (flair, phase, lesion_index, etc.)."""
        return nib.load(str(self.dir / f"{name}.nii.gz"))
```

### 1B. Experiment slimmed down

With Dataset owning path resolution, Experiment's `cases` property delegates to Dataset and adds inference paths (which DO depend on `run_dir`):

```python
class Experiment:
    @cached_property
    def cases(self) -> pd.DataFrame:
        """Dataset cases augmented with inference paths for this run."""
        df = self.dataset.cases.copy()
        df["inference"] = df.apply(self._find_inference_path, axis=1)
        return df
```

**Methods to remove from Experiment:**
- `evaluate()` — replaced by `analysis.metrics.performance` (deprecate with warning first)
- `cases_df` property — `cases` IS a DataFrame now
- `get_case()` — use `cases.loc[(subid, lesion_index)]`
- `subject_dir()` (line 152-153, currently broken — returns `self.datalist`)
- `_build_cases()` — logic moves to `Dataset._build_cases_df()`

**Methods that stay on Experiment:** `setup()`, `train()`, `predict()`, `from_run_dir()`, `cleanup()`, `has_trained()`, `create_rois()`, `prepare_data()`

### 1C. Expand AlgoConfig to cover full hyper_parameters.yaml

**Problem**: AlgoConfig covers ~60% of SegResNet's hyper_parameters.yaml. Missing fields like `modality`, `cache_rate`, `resample`, `normalize_mode`, `sigmoid`, phase configs (finetune/validate/infer), and execution flags (debug, ckpt_save, etc.).

**Add missing fields to AlgoConfig:**
```python
@attrs.define()
class AlgoConfig:
    # --- Data ---
    modality: str = "mri"
    cache_rate: float | None = None

    # --- Spatial (new) ---
    resample: bool = False
    resample_resolution: list[float] | None = None
    normalize_mode: str = "meanstd"
    intensity_bounds: list[float] | None = None
    sigmoid: bool = False
    orientation_ras: bool = True

    # --- Execution (new) ---
    debug: bool = False
    ckpt_save: bool = True
    validate_final_original_res: bool = True
    calc_val_loss: bool = False

    # --- Phases (None = use template default) ---
    finetune: dict | None = None
    validate_phase: dict | None = None  # "validate" conflicts with method name
    infer: dict | None = None

    # --- Network (nested dict, None = template default) ---
    network: dict | None = None

    # ... existing fields stay ...
```

**Add `from_template()` classmethod:**
```python
@classmethod
def from_template(cls, algo: str = "segresnet") -> AlgoConfig:
    """Create config with defaults from the algorithm template's hyper_parameters.yaml."""
    template_path = PROJECT_ROOT / "training" / "algorithm_templates" / algo / "configs" / "hyper_parameters.yaml"
    return cls.load_from_yaml(template_path)
```

**Register SwinUNETR and DiNTS stubs** in `_ALGO_REGISTRY` (implementation deferred, but slots reserved):
```python
@attrs.define
class SwinUNETRConfig(AlgoConfig):
    algo: str = "swinunetr"
    use_pretrain: bool = True
    pretrained_path: str | None = None
    lr_scheduler: dict | None = None
    adapt_valid_mode: bool = True
    early_stop_mode: bool = True
    early_stop_patience: int = 5

@attrs.define
class DiNTSConfig(AlgoConfig):
    algo: str = "dints"
    search_config: dict | None = None  # entire "searching:" block
```

### ABCs vs Simple Inheritance (answering CLAUDE.md question)

**Recommendation: don't use ABCs here.**

ABCs enforce "you must implement method X" at instantiation time. They're valuable when:
- Multiple independent implementations satisfy a shared interface (like `collections.abc.Mapping`)
- External code depends on the interface contract
- You want to catch missing methods early

Here, the algorithm configs are **data containers with slightly different fields**, not polymorphic implementations of a behavioral interface. The `_ALGO_REGISTRY` dispatch in `from_dict()` already handles "which class do I instantiate?" — and `to_input_dict()` works via attrs field iteration, not abstract methods that each subclass overrides differently.

ABCs would add ceremony (`from abc import ABC, abstractmethod`, `@abstractmethod` decorators) without catching real bugs. The risk they guard against — someone forgetting to implement a method — doesn't apply when the base class methods already work for all subclasses (SegResNet, SwinUNETR, DiNTS all use the same `to_input_dict()` pattern with minor remapping).

**Simple inheritance with a concrete base class** (the current pattern) is the right call. The improvement is making AlgoConfig's fields comprehensive, not adding abstraction layers.

---

## Phase 2: Analysis Pipeline Redesign — `src/analysis/`

### New directory structure

```
src/analysis/
    __init__.py           # Public API
    _cache.py             # Cache layer (load/save/invalidate)
    loaders.py            # Data loading: load_run_data()
    compile.py            # compile_all_metrics, compile_experiment_metrics, compile_grid_metrics
    metrics/
        __init__.py       # Re-export metric functions
        confusion.py      # get_confusion_matrix, compute_derived_metrics, compute_casewise_stats
        mlflow.py         # analyze_unified_mlruns, aggregate_metrics, mlflow_metrics()
        performance.py    # performance_metrics()
        display.py        # format_param_value, rename_metric, order_columns
```

### Key design: Separate loading, computation, and caching

**`loaders.py`** — Replaces `load_or_cache_run()` monolith:

```python
def load_run_data(
    experiment: Experiment,
    compute_performance: bool = True,
    compute_mlflow: bool = True,
) -> dict:
    """Load all raw data for a run. Each section independently toggleable.

    Returns dict with keys: cases, case_performance, mlflow_fold_data,
    mlflow_aggregated, hyper_params
    """
```

**`_cache.py`** — Orthogonal caching:

```python
def cache_run(run_dir: Path, data: dict) -> Path:
    """Write run data to cache."""

def load_cached(run_dir: Path) -> dict | None:
    """Load cached data if exists and fresh."""

def cached_load_run_data(experiment, use_cache=True, **kwargs) -> dict:
    """load_run_data with caching wrapper."""
```

### Composable metric functions

Each metric function has the same signature: `(run_data: dict, **options) -> dict`

This makes them composable — define a new metric function, pass it to `compile_all_metrics()`:

```python
# Existing:
def performance_metrics(run_data: dict, splits=["testing"]) -> dict: ...
def mlflow_metrics(run_data: dict) -> dict: ...

# User-defined composition:
def fp_per_100_cases(run_data: dict) -> dict:
    cases = [c for c in run_data["case_performance"] if c["split"] == "testing"]
    total_fp = sum(c["fp"] for c in cases)
    return {"fp_per_100": (total_fp / len(cases) * 100) if cases else None}
```

### Migration map

| Current file | What moves where | What gets deleted |
|---|---|---|
| `scripts/compute_performance_metrics.py` | `get_confusion_matrix`, `compute_derived_metrics`, `compute_casewise_stats` → `analysis/metrics/confusion.py` | `analyze_dataset()`, CLI `main()` |
| `scripts/analyze_mlflow_runs.py` | `analyze_unified_mlruns`, `aggregate_metrics`, `load_metrics_from_file` → `analysis/metrics/mlflow.py` | CLI `main()`, print/plot helpers → `display.py` |
| `scripts/compile_run_metrics.py` | `performance_metrics` → `analysis/metrics/performance.py`; `mlflow_metrics` → `analysis/metrics/mlflow.py`; `compile_*` functions → `analysis/compile.py`; display helpers → `analysis/metrics/display.py` | `load_or_cache_run()` replaced by `loaders.py` + `_cache.py`; entire file consumed |

---

## Phase 3: Image Analysis Module — `src/analysis/image/`

### New directory structure

```
src/analysis/image/
    __init__.py
    geometry.py           # get_convex_hull, rim_convex_hull_volume, rim_enclosing_sphere_radius
    lesion_analysis.py    # analyze_prl_case, _get_lesion_rim, get_center_label, _crop_from_volume, _parse_bounding_boxes
    plotting.py           # plot_lesion_rim_3d
```

### Migration from `scripts/lesion_diagnostics.py`

- Pure geometry: `get_convex_hull`, `rim_convex_hull_volume`, `rim_enclosing_sphere_radius` → `geometry.py`
- Lesion-specific: `_parse_bounding_boxes`, `_crop_from_volume`, `_get_center_lesion`, `get_center_label`, `_get_lesion_rim`, `_count_rim_for_lesion`, `count_predicted_prls`, `analyze_prl_case` → `lesion_analysis.py`
- Visualization: `plot_lesion_rim_3d` → `plotting.py`
- Delete: `rim_enclosing_sphere_radius0` (duplicate), `run_diagnostics`, `print_diagnostics`, CLI `__main__` block

### Refactor `analyze_prl_case` to use new core classes

Currently takes `(prl_case: dict, experiment, bbox_suffix)`. Refactor to accept a `Subject` or just use Dataset directly:

```python
def analyze_prl_case(case: pd.Series, dataset: Dataset) -> tuple[dict, dict]:
    """Analyze a single PRL case.
    case: row from dataset.cases or experiment.cases
    """
```

---

## Phase 4: Cleanup and Integration

### 4A. Update imports across codebase

- `src/cli.py`: Update `prl metrics` command to use `analysis.metrics.performance`
- Notebooks: Update imports from `scripts.*` → `analysis.*`
- Add temporary backward-compat re-exports in `scripts/` files:
  ```python
  # scripts/compute_performance_metrics.py (temporary)
  from analysis.metrics.confusion import *  # noqa: backward compat
  ```

### 4B. Clean up old scripts

After notebooks/CLI are updated:
- `scripts/compute_performance_metrics.py` — reduce to thin compat shim or delete
- `scripts/analyze_mlflow_runs.py` — reduce to thin compat shim or delete
- `scripts/compile_run_metrics.py` — delete (fully consumed)
- `scripts/lesion_diagnostics.py` — delete (fully consumed)

### 4C. Update `pyproject.toml`

Add `analysis` to packages list.

### 4D. Update CLAUDE.md

Update directory layout section to reflect new `src/analysis/` structure.

---

## Implementation Order

```
Phase 1A: Dataset.preprocess + cases DataFrame
    ↓
Phase 1B: Experiment slimdown (depends on 1A)
    ↓
Phase 1C: AlgoConfig expansion (independent of 1A/1B but logically grouped)
    ↓
Phase 2: analysis/metrics/ + loaders + cache + compile  ←  can start after 1B
    ↓                                                       (needs Experiment.cases)
Phase 3: analysis/image/  ←  can run parallel with Phase 2
    ↓
Phase 4: Cleanup (after 2 + 3 complete)
```

**Estimated scope per phase:**
- Phase 1: ~3 files modified (dataset.py, experiment.py, configs.py)
- Phase 2: ~7 new files, 3 old files migrated
- Phase 3: ~3 new files, 1 old file migrated
- Phase 4: Import updates + deletions
