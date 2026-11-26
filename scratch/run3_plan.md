# Plan: roi_train2 run3 — Parameter Selection for Overnight Run

## Context: Comparing Past Runs

| Metric | roi_train1 | roi_train2/run1 | roi_train2/run2 |
|--------|-----------|-----------------|-----------------|
| **val/acc** | **0.460** | 0.425 | **0.470** |
| **val_class/acc_1 (RIM)** | **0.323** | 0.178 | 0.302 |
| val_class/acc_0 (lesion) | 0.597 | 0.673 | 0.638 |
| train/loss | 0.641 | 0.769 | 0.637 |
| expand_z | 2 | 8 | 2 |
| learning_rate | 0.0002 | 0.0004 | 0.0002 |
| num_images_per_batch | 1 | 32 | 1 |
| num_epochs | 250 | 300 | 250 |
| roi_size | [44,44,8] | [40,40,16] | [44,44,8] |
| blocks_down | [1,2,4] | [1,2,2,4] | [1,2,4] |

## Analysis

**run2 actually did better overall** (0.470 vs 0.460 val/acc) but **slightly worse on the rim class** (0.302 vs 0.323). The rim class is what matters most for PRL detection, so this is why it feels like a regression.

**run1 was clearly worse** — the aggressive parameters (2x learning rate, batch=32, deeper network, expand_z=8) hurt badly. The larger z-expansion diluted the rim signal with more background.

**The core problem is class imbalance.** From the fold analysis: rim makes up only ~0.4-0.9% of foreground voxels in PRL cases, and most cases are lesion-only (no rim at all). The model is learning lesion well but struggling with rim because it's so rare.

## Recommendation for Tonight's Run

**Change one thing: increase `num_epochs` to 500.**

Why:
- run2's train loss (0.637) is still meaningfully above run1's (which ran 300 epochs and got to 0.769 — but that was with broken params). The loss is still dropping, suggesting more room to learn
- This is the safest parameter to increase — it can't make things worse (you just look at the best checkpoint)
- roi_train1 and run2 used identical params except for the data. If the same params with more/better data gave slightly worse rim accuracy, the model may just need more time to learn the rare rim features
- The run will take longer but it's overnight anyway
- Changing LR or architecture introduces more confounding variables

## What to Do

In `training/roi_train2/`, create a `run3` monai_config.json:

```json
{
    "training_work_home": "${TRAIN_ROOT}/roi_train2",
    "N_FOLDS": 5,
    "TEST_SPLIT": 0.2,
    "train_param": {
        "algos": ["segresnet"],
        "learning_rate": 0.0002,
        "num_images_per_batch": 1,
        "num_epochs": 500,
        "num_warmup_epochs": 1,
        "num_epochs_per_validation": 1,
        "roi_size": [44, 44, 8]
    }
}
```

Then launch:

```bash
python training/roi_train2/train.py --run-dir /media/smbshare/srs-9/prl_project/training/roi_train2/run3
```

## What to Try After This

For future runs (grid search territory), the parameters that would likely have the biggest impact on rim detection:

1. **Class weighting / loss function** — force the model to pay more attention to the rare rim class (this is the real lever for class imbalance)
2. **roi_size increase** — try [56, 56, 12] to capture more spatial context around rims
3. **Learning rate schedule** — try cosine annealing or a lower LR (0.0001) for finer boundary learning
4. **expand_xy** — try 30 to give more context around lesions (but be careful: run1 showed that too much context hurts)
