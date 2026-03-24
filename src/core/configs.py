"""Configuration dataclasses for the PRL pipeline.

PreprocessingConfig: ROI expansion parameters and execution options.
AlgoConfig: Base training hyperparameters for any MONAI Auto3DSeg algorithm.
SegResNetConfig: SegResNet-specific network architecture params.
"""

from __future__ import annotations

from typing import ClassVar

import attrs


@attrs.define(frozen=True)
class PreprocessingConfig:
    """ROI cropping and data preparation parameters.

    Frozen (immutable + hashable) so identical configs can be deduplicated
    when sweeping expand_xy/expand_z in an HPO grid.
    """

    expand_xy: int = attrs.field(default=20, validator=attrs.validators.ge(0))
    expand_z: int = attrs.field(default=2, validator=attrs.validators.ge(0))
    images: tuple[str, ...] = attrs.field(
        default=("flair", "phase"), converter=tuple,
    )
    processes: int | None = None  # None = sequential
    dry_run: bool = False

    @property
    def image_prefix(self) -> str:
        """Joined sorted image names, e.g. 'flair.phase'."""
        return ".".join(sorted(self.images))

    @property
    def suffix(self) -> str:
        """The xy/z suffix used in filenames, e.g. 'xy20_z2'."""
        return f"xy{self.expand_xy}_z{self.expand_z}"

    @property
    def datalist_suffix(self) -> str:
        """Combined image + expansion suffix, e.g. 'flair.phase_xy20_z2'."""
        return f"{self.image_prefix}_{self.suffix}"


# ---------------------------------------------------------------------------
# Algorithm training configs
# ---------------------------------------------------------------------------

# Registry for algo name → config subclass (populated after class definitions)
_ALGO_REGISTRY: dict[str, type[AlgoConfig]] = {}


@attrs.define
class AlgoConfig:
    """Base training parameters shared across MONAI Auto3DSeg algorithms.

    Fields mirror the user-tunable entries in hyper_parameters.yaml.
    Adding a field here automatically flows through to_input_dict() and
    to_monai_config_dict() — no manual enumeration needed.

    Fields set to None are omitted from the AutoRunner input dict, letting
    the algorithm template / auto-scaling decide the value.
    """

    algo: str = "segresnet"

    # --- Training schedule ---
    learning_rate: float = 0.0002
    num_epochs: int = 500
    num_images_per_batch: int = 1
    batch_size: int | None = None
    num_warmup_epochs: int = 1
    num_epochs_per_validation: int = 1
    num_epochs_per_saving: int = 1
    num_crops_per_image: int = 1
    num_workers: int = 4

    # --- ROI / spatial ---
    roi_size: list[int] = attrs.Factory(lambda: [44, 44, 8])
    crop_ratios: list[int] | None = None
    crop_mode: str = "ratio"
    crop_foreground: bool = True

    # --- Auto-scaling ---
    auto_scale_allowed: bool = True
    auto_scale_batch: bool = True
    auto_scale_roi: bool = False
    auto_scale_filters: bool = False

    # --- Execution ---
    amp: bool = True
    channels_last: bool = True
    determ: bool = False
    early_stopping_fraction: float = 0.001

    # --- Loss / optimizer (nested dicts, None = template default) ---
    loss: dict | None = None
    optimizer: dict | None = None

    # --- Pass-through for rare/future params ---
    extra: dict = attrs.Factory(dict)

    # Fields excluded from MONAI input dict (metadata only)
    _SKIP_IN_INPUT: ClassVar[frozenset] = frozenset({"algo", "extra"})

    def to_input_dict(self, datalist_path, dataroot) -> dict:
        """Build the AutoRunner input dict.

        ALL training params go here. Everything flows through
        fill_template_config() → hyper_parameters.yaml.
        """
        d = {
            "modality": "MRI",
            "datalist": str(datalist_path),
            "dataroot": str(dataroot),
        }
        for field in attrs.fields(type(self)):
            if field.name in self._SKIP_IN_INPUT:
                continue
            val = getattr(self, field.name)
            if val is not None:
                d[field.name] = val
        d.update(self.extra)
        return d

    def to_label_config_dict(self, preprocess: PreprocessingConfig, dataset) -> dict:
        """Generate the label_config.json dict for a run directory."""
        return {
            "dataroot": str(dataset.data_root),
            "train_home": str(dataset.train_home),
            "dataset_name": dataset.name,
            "prl_df": str(dataset.prl_df_path),
            "subjects": str(dataset.subjects_path),
            "suffix_to_use": str(dataset.suffix_to_use_path),
            "expand_xy": preprocess.expand_xy,
            "expand_z": preprocess.expand_z,
            "images": list(preprocess.images),
        }

    def to_monai_config_dict(self, dataset) -> dict:
        """Generate the monai_config.json dict for a run directory.

        Includes ALL fields (even None) for complete round-trip reconstruction.
        """
        train_param = {"algo": self.algo}
        for field in attrs.fields(type(self)):
            if field.name in self._SKIP_IN_INPUT:
                continue
            train_param[field.name] = getattr(self, field.name)
        train_param.update(self.extra)
        return {
            "training_work_home": str(dataset.work_home),
            "N_FOLDS": dataset.n_folds,
            "TEST_SPLIT": dataset.test_split,
            "train_param": train_param,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AlgoConfig:
        """Create from a dict (dataset.yaml, monai_config.json, etc.).

        Dispatches to the correct subclass based on the 'algo' or 'algos' key.
        Known fields become typed attrs; everything else goes to 'extra'.
        """
        d = dict(d)  # shallow copy to pop from

        # Handle both "algo": "segresnet" and legacy "algos": ["segresnet"]
        algo = d.pop("algo", None)
        if algo is None:
            algos = d.pop("algos", ["segresnet"])
            algo = algos[0] if isinstance(algos, list) else algos

        config_cls = _ALGO_REGISTRY.get(algo, cls)
        known = attrs.fields_dict(config_cls)
        known_params = {k: v for k, v in d.items() if k in known and k != "extra"}
        extra_params = {k: v for k, v in d.items() if k not in known}
        return config_cls(algo=algo, **known_params, extra=extra_params)


@attrs.define
class SegResNetConfig(AlgoConfig):
    """SegResNet-specific parameters.

    Network architecture fields are remapped to MONAI's 'network#' nested-key
    syntax in to_input_dict(). None = let MONAI auto_adjust decide.
    """

    algo: str = "segresnet"

    # Network architecture
    init_filters: int | None = None
    blocks_down: list[int] | None = None
    dsdepth: int | None = None
    norm: str | None = None

    _NETWORK_KEY_MAP: ClassVar[dict[str, str]] = {
        "init_filters": "network#init_filters",
        "blocks_down": "network#blocks_down",
        "dsdepth": "network#dsdepth",
        "norm": "network#norm",
    }

    def to_input_dict(self, datalist_path, dataroot) -> dict:
        d = super().to_input_dict(datalist_path, dataroot)
        # Remap network fields to MONAI's nested-key syntax
        for attr_name, monai_key in self._NETWORK_KEY_MAP.items():
            if attr_name in d:
                d[monai_key] = d.pop(attr_name)
        return d


# Register subclasses
_ALGO_REGISTRY["segresnet"] = SegResNetConfig

# Backward-compatible alias
TrainingConfig = AlgoConfig
