# Changes

If you can see whats on branch pipeline-v2, you can see the files im referencing. I used the old paradigm to run stage2 tho. But essentially, the experiment config was:

```json
{
  "train_home": "${PROJECT_ROOT}/training/roi_train2",
  "experiment_name": "stage2_numcrops_incbackground_sweep",
  "base_label_config": "label_config.jsonc",
  "base_monai_config": "monai_config.jsonc",
  "param_grid": {
    "monai": {
      "train_param.num_crops_per_image": [2, 4, 8],
      "train_param.loss": [
        {
          "_target_": "DiceCELoss",
          "include_background": true,
          "squared_pred": true,
          "smooth_nr": 0,
          "smooth_dr": 1.0e-05,
          "softmax": "$not @sigmoid",
          "sigmoid": "$@sigmoid",
          "to_onehot_y": "$not @sigmoid"
        },
        {
          "_target_": "DiceCELoss",
          "include_background": false,
          "squared_pred": true,
          "smooth_nr": 0,
          "smooth_dr": 1.0e-05,
          "softmax": "$not @sigmoid",
          "sigmoid": "$@sigmoid",
          "to_onehot_y": "$not @sigmoid"
        }
      ]
    }
  }
}
```

In order to for roi_train2/train.py to handle the dicts, I made made a small edit:

```python
for key in list(train_param):
    if isinstance(train_param[key], list) or isinstance(train_param[key], dict):
        input_dict[key] = train_param.pop(key)
        logger.info(f"{key}: {input_dict[key]}")

runner = AutoRunner(
    work_dir=work_dir,
    algos=algos,
    input=input_dict,
    mlflow_tracking_uri=mlflow_tracking_uri,
    mlflow_experiment_name=mlflow_experiment_name,
)
for key in train_param:
    logger.info(f"{key}: {train_param[key]}")
runner.set_training_params(train_param)
```

but shit you're right, the num_crops per image did not survive, all of them ended up being 2. But i dont know what they would have copied from: batch_size and num_images_per_batch were both 1, not 2. Where would it have gotten that from? At least include background survived. can you help me adapt the new paradigm to work with all this and have num_crops_per_image survive?

```yaml
_meta_: {}
bundle_root: /home/shridhar.singh9-umw/prl_project/training/roi_train2/stage2_numcrops_incbackground_sweep/run4/segresnet_2
ckpt_path: $@bundle_root + '/model'
mlflow_tracking_uri: /home/shridhar.singh9-umw/prl_project/training/roi_train2/stage2_numcrops_incbackground_sweep/run4/mlruns
mlflow_experiment_name: run4
data_file_base_dir: /home/shridhar.singh9-umw/prl_project/data
data_list_file_path: /home/shridhar.singh9-umw/prl_project/training/roi_train2/stage2_numcrops_incbackground_sweep/run4/datalist_xy20_z2.json
modality: mri
fold: 2
input_channels: 2
output_classes: 3
class_names: null
class_index: null
debug: false
ckpt_save: true
cache_rate: null
roi_size: [44, 44, 8]
auto_scale_allowed: true
auto_scale_batch: true
auto_scale_roi: false
auto_scale_filters: false
quick: false
channels_last: true
validate_final_original_res: true
calc_val_loss: false
amp: true
log_output_file: null
cache_class_indices: null
early_stopping_fraction: 0.001
determ: false
orientation_ras: true
crop_foreground: true
learning_rate: 0.0002
batch_size: 1
num_images_per_batch: 1
num_epochs: 500
num_warmup_epochs: 1
sigmoid: false
resample: false
resample_resolution: [0.7999979257583618, 0.800000011920929, 0.7999999964916801]
crop_mode: ratio
normalize_mode: meanstd
intensity_bounds: [801.0101143671224, 1419.8391564510932]
num_epochs_per_validation: 1
num_epochs_per_saving: 1
num_workers: 4
num_steps_per_image: null
num_crops_per_image: 2
loss: {_target_: DiceCELoss, include_background: false, squared_pred: true, smooth_nr: 0,
  smooth_dr: 1.0e-05, softmax: $not @sigmoid, sigmoid: $@sigmoid, to_onehot_y: $not
    @sigmoid}
optimizer: {_target_: torch.optim.AdamW, lr: '@learning_rate', weight_decay: 1.0e-05}
network:
  _target_: SegResNetDS
  init_filters: 32
  blocks_down: [1, 2, 4]
  norm: INSTANCE_NVFUSER
  in_channels: '@input_channels'
  out_channels: '@output_classes'
  dsdepth: 4
finetune: {enabled: false, ckpt_name: $@bundle_root + '/model/model.pt'}
validate: {enabled: false, ckpt_name: $@bundle_root + '/model/model.pt', output_path: $@bundle_root
    + '/prediction_validation', save_mask: false, invert: true}
infer: {enabled: false, ckpt_name: $@bundle_root + '/model/model.pt', output_path: $@bundle_root
    + '/prediction_' + @infer#data_list_key, data_list_key: testing}
anisotropic_scales: false
spacing_median: [0.7999979257583618, 0.800000011920929, 0.800000011920929]
spacing_lower: [0.7999419303265637, 0.7999999777833893, 0.7999999709715957]
spacing_upper: [0.8000359182650915, 0.8000001065156629, 0.8000000891649375]
image_size_mm_median: [35.200282318776104, 35.999999000252515, 6.400000183340294]
image_size_mm_90: [40.800152599811554, 42.16000045588806, 12.800000190734863]
image_size: [51, 52, 16]
crop_ratios: null
```

