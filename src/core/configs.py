"""Configuration dataclasses for the PRL pipeline.

PreprocessingConfig: ROI expansion parameters and execution options.
TrainingConfig: MONAI Auto3DSeg / SegResNet training hyperparameters.
"""

from __future__ import annotations

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


#TODO: a workaround added to handle new parameters; 
@attrs.define
class TrainingConfig:
    """MONAI AutoRunner training hyperparameters.

    These map directly to the train_param dict consumed by AutoRunner.
    """
    batch_size: int = None
    num_crops_per_image: int = 1
    auto_scale_allowed: bool = True
    auto_scale_batch: bool = False
    algos: list[str] = attrs.Factory(lambda: ["segresnet"])
    learning_rate: float = 0.0002
    num_images_per_batch: int = 1
    num_epochs: int = 500
    num_warmup_epochs: int = 1
    num_epochs_per_validation: int = 1
    roi_size: list[int] = attrs.Factory(lambda: [44, 44, 8])
    crop_ratios: list[int] | None = None
    loss: dict | None = None  # Full DiceCELoss config, or None for MONAI default
    extra: dict = attrs.Factory(dict)  # Pass-through for any additional MONAI params

    def to_input_dict(self, datalist_path, dataroot) -> dict:
        """Build the AutoRunner input dict. ALL training params go here.

        Everything flows through fill_template_config() → config.update(input_config)
        → hyper_parameters.yaml. No need for set_training_params() / CLI overrides.
        """
        d = {
            "modality": "MRI",
            "datalist": str(datalist_path),
            "dataroot": str(dataroot),
            "learning_rate": self.learning_rate,
            "num_images_per_batch": self.num_images_per_batch,
            "num_epochs": self.num_epochs,
            "num_warmup_epochs": self.num_warmup_epochs,
            "num_epochs_per_validation": self.num_epochs_per_validation,
            "roi_size": self.roi_size,
            "batch_size": self.batch_size,
            "auto_scale_allowed": False,
            "auto_scale_batch": False,
            "num_crops_per_image": 1,
        }
        if self.crop_ratios is not None:
            d["crop_ratios"] = self.crop_ratios
        if self.loss is not None:
            d["loss"] = self.loss
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
        """Generate the monai_config.json dict for a run directory."""
        train_param = {
            "algos": self.algos,
            "learning_rate": self.learning_rate,
            "num_images_per_batch": self.num_images_per_batch,
            "num_epochs": self.num_epochs,
            "num_warmup_epochs": self.num_warmup_epochs,
            "num_epochs_per_validation": self.num_epochs_per_validation,
            "roi_size": self.roi_size,
            "crop_ratios": self.crop_ratios,
            "batch_size": self.batch_size
        }
        if self.loss is not None:
            train_param["loss"] = self.loss
        train_param.update(self.extra)
        return {
            "training_work_home": str(dataset.work_home),
            "N_FOLDS": dataset.n_folds,
            "TEST_SPLIT": dataset.test_split,
            "train_param": train_param,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        """Create from a dict (e.g. parsed from dataset.yaml defaults.training).

        Known fields are mapped to typed attrs fields. Anything else is collected
        into `extra` and passed through to to_input_dict() verbatim, so new MONAI
        params can be added to dataset.yaml without touching this class.
        """
        known = attrs.fields_dict(cls)
        known_params = {k: v for k, v in d.items() if k in known and k != "extra"}
        extra_params = {k: v for k, v in d.items() if k not in known}
        return cls(**known_params, extra=extra_params)
