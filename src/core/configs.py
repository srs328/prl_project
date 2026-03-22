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


@attrs.define
class TrainingConfig:
    """MONAI AutoRunner training hyperparameters.

    These map directly to the train_param dict consumed by AutoRunner.
    """

    algos: list[str] = attrs.Factory(lambda: ["segresnet"])
    learning_rate: float = 0.0002
    num_images_per_batch: int = 1
    num_epochs: int = 500
    num_warmup_epochs: int = 1
    num_epochs_per_validation: int = 1
    roi_size: list[int] = attrs.Factory(lambda: [44, 44, 8])
    crop_ratios: list[int] | None = None

    def to_train_param(self) -> dict:
        """Convert to the dict format AutoRunner.set_training_params() expects.

        List-valued params are excluded here (they go into the input dict instead)
        to avoid BundleAlgo CLI mangling.
        """
        param = {
            "learning_rate": self.learning_rate,
            "num_images_per_batch": self.num_images_per_batch,
            "num_epochs": self.num_epochs,
            "num_warmup_epochs": self.num_warmup_epochs,
            "num_epochs_per_validation": self.num_epochs_per_validation,
        }
        if self.crop_ratios is not None:
            param["crop_ratios"] = self.crop_ratios
        return param

    def to_input_dict(self, datalist_path, dataroot) -> dict:
        """Build the AutoRunner input dict.

        List-valued params (roi_size, crop_ratios) are placed here instead of
        train_param to avoid Fire CLI mangling in BundleAlgo.train().
        """
        input_dict = {
            "modality": "MRI",
            "datalist": str(datalist_path),
            "dataroot": str(dataroot),
            "roi_size": self.roi_size,
        }
        if self.crop_ratios is not None:
            input_dict["crop_ratios"] = self.crop_ratios
        return input_dict

    def to_label_config_dict(self, preprocess: PreprocessingConfig, dataset) -> dict:
        """Generate the label_config.json dict for a run directory."""
        return {
            "dataroot": str(dataset.data_root),
            "train_home": str(dataset.source_home),
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
        }
        return {
            "training_work_home": str(dataset.work_home),
            "N_FOLDS": dataset.n_folds,
            "TEST_SPLIT": dataset.test_split,
            "train_param": train_param,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        """Create from a dict (e.g. parsed from dataset.yaml defaults.training)."""
        return cls(**{k: v for k, v in d.items() if k in attrs.fields_dict(cls)})
