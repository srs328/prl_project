# PRL Project — Where Things Stand (Nov 2026)

Catch-up summary written 2026-11-13 BY [Claude](https://claude.ai/chat/18345581-241a-421a-8bae-37b9add08b22) from the `refactor-analysis-and-core` branch: notes, notebooks, result CSVs and
commit history. I could only see what's in git; anything on the HPC or `/media/smbshare` that isn't committed is
marked **(verify)**.

---

## TL;DR

- **Pipeline:** LST-AI lesion index → per-lesion ROI crops (flair+phase) → SegResNet (Auto3DSeg, 5-fold) segments
  lesion (1) and rim (2) → lesion-level logistic regression on shape/radiomic features filters out false-positive rims →
  uncropped full-brain PRL map.
- **Working model:** `roi_train2/stage6_sweep_dicece_wts/run1` (DiceCE weights `[1,1,10]`). Rim Dice on PRL test
  cases is about 0.60 ± 0.12 (n = 9 PRL test cases). Rim weighting was the big win; later sweeps are within noise.
- **Classifier:** ROC-AUC about 0.86–0.88. It improves full-brain rim Dice from 0.60 to 0.63 and HD95 from 35 to 15, but both
  numbers are optimistic (lesion-level split, refit on all data). Needs subject-grouped CV before it can be quoted.
- **Code:** The refactor in `plan.md` is mostly **done**: `Dataset.cases`, `Subject`, the expanded `AlgoConfig`, the
  `src/analysis/` package, and the CLI with `infer` and `classify`. The old `scripts/*` are now shims, but notebooks still import from them.
- **Last activity:** Real analysis work stopped around **late May / early June**. In **August** you wrote
  implementation-agnostic specs for a possible rewrite. Today's commit re-ran `evaluate_inference2.ipynb`.

---

## Timeline (from commits)

| When | What happened |
|---|---|
| Mar 19–20 | Metrics tooling. Fold 4 diagnosed as a lesion-size distribution shift. MLflow unified per run. |
| Mar 23–25 | OOP core (`Dataset` / `Experiment` / `ExperimentGrid`) and HPC grid launching. Found the `num_crops` override bug. Stages 1–3 run. |
| Mar 25–27 | Stages 4–8. PRL lesion analysis (rim geometry). First logistic-regression classifier for a conference abstract. Inter-rater notes. |
| Mar 29 | Started the `prlsegresnet` MONAI bundle (standalone, for MonaiLabel). |
| May 4–22 | Back after a break. Inference on unlabeled subjects (`inference_dataset`, `full_inference_dataset`). Poster/presentation figures. |
| May 30 – Jun 1 | Pyradiomics features, classifier reorganization, `prl classify` CLI. Pipeline_Notes updated with audit goals. |
| Aug 8 | Wrote `notes/Implementation Agnostic Plan/` specs (v1, v2, revised) and `AGENTS.md`. Added subject category lists. No training. |
| Nov 13 | Re-ran `evaluate_inference2.ipynb`. |

---

## The data (three subject sets)

| Set (code name) | n | What it has | Used for |
|---|---|---|---|
| `roi_train2` ("training") | 24 subjects, ~195 test-split cases, **9 PRL test cases** | Voxelwise rim labels (`prl_mask_def_prob_<rater>`) | Segmentation training and evaluation |
| `inference_dataset` | 52 subjects, ~3,900 lesions | Which lesion index is a PRL (from the spreadsheet), no rim voxels | Classifier training and validation |
| `full_inference_dataset` | ~477 subjects | Subject-level PRL counts only | Count comparison and inference |

Label notes: only "definite" and "probable" PRLs are used; "possible" is excluded. The rater suffix per subject is in
`labels_to_use.csv` (seniority is CH > SRS > LR, but you assign suffixes by hand). Images: flair and phase. T1 was tried as a third channel.

---

## Segmentation results (rim Dice on PRL test cases, mean ± SD)

Full table: `notebooks/hpo_performance_metrics.csv`. Grid definitions: `training/roi_train2/experiment_config_stage*.json`.

| Stage | What was varied | Rim Dice range | Takeaway |
|---|---|---|---|
| pilot `roi_train1` | old labels | 0.36 | — |
| `run1`–`run3` | baseline | 0.41–0.55 | — |
| stage1 crop_lr | crop_ratios × lr | 0.50–0.55 | Oversampling didn't help. lr 2e-4 is fine. |
| stage2 numcrops_dicece | background on/off × weight | 0.51 → **0.59** | **Weight `[1,1,5]` is the big jump.** Background doesn't matter. |
| stage3 | num_crops 1 vs 4 × background | 0.58–0.60 | No real difference. |
| stage4/5 | lesion weight 2–5, batch | 0.52–0.58 | **Raising the lesion weight hurts.** |
| stage6 | rim weight 10/15 | 0.58–**0.604** | `run1` [1,1,10] chosen as the working model. |
| stage7 expand | crop expansion | 0.57–0.60 | **(verify)** which expand values were used. |
| stage8 push_epochs | 1000 epochs, 4 crops | **0.613** | Best number, but within noise of stage6. |
| `_t1` runs | flair+phase+t1 | 0.586–0.587 | Same Dice. Downstream classification "much worse" (unexplained). |

**Things to remember**
- With 9 PRL test cases and SD ~0.12, the top runs (0.59–0.61) are statistically indistinguishable.
- The `num_crops_per_image` and `batch_size` you set aren't necessarily what ran. SegResNet's auto-scaling rewrites them, and
  when num_crops > 1 it also cuts `num_epochs` (e.g. 500 → 166). Since stage 5, auto-scaling has been disabled in the grid configs.
  Check the resolved `hyper_parameters.yaml` in each run directory.
- Voxel-level confusion counts treat label 2 = positive and label 1 = negative (background is ignored). Precision sits around 0.42–0.45
  across all runs, so false-positive rim voxels are the persistent problem. That's why the classifier stage exists.

---

## Classifier stage

- **Code:** `src/analysis/image/classify.py` (logistic regression, `prepare_data` / `train_model` / `evaluate_model`), lesion
  features in `src/analysis/image/lesion_analysis.py` (`analyze_prl_case`, `screen_for_iron`), and pyradiomics through
  `notebooks/radiomics_worker.py`. That script runs in a separate Python 3.6 conda env (`pyradiomics2`) called through a subprocess.
- **Features:** rim volume, hull volume, PCA size/sphericity/planarity/linearity, radius mean/std, and GLCM texture.
  `rim_volume_infer` is the dominant coefficient.
- **Results:** test ROC-AUC 0.86–0.88; PRL precision 0.67–0.75, recall 0.77–0.80 (~40 PRL test lesions).
- **Full-brain effect on the 24 training subjects:** Dice 0.600 → 0.632, precision 0.50 → 0.57, recall 0.77 → 0.74, HD95 35 → 15.
- **Caveats to fix before reporting:**
  1. The train/test split is per lesion (stratified by label) and **not grouped by subject**, so the same subject's lesions can
     appear on both sides.
  2. The before/after Dice uses a model refit with `test_size=0`. Your notebook comment calls this "double dipping".
- **Outputs:** derived CSVs live in the top-level `analysis/` folder, named `prl_<dataset>_image_stats-<run_id>.csv`. That
  naming is good provenance; keep it. Some intermediates are in `notebooks/tmp/`.

---

## Code state

**Branch:** `refactor-analysis-and-core`

**Done (per `plan.md`):**
- Phase 1: `Dataset.preprocess`, `Dataset.cases` (a DataFrame with resolved paths), `Subject`, `Experiment.cases` adds an
  `inference` column, `AlgoConfig.from_template()`, and SwinUNETR/DiNTS config stubs.
- Phases 2–3: the `src/analysis/` package (`loaders`, `_cache`, `compile`, `metrics/`, `image/`).
- CLI: `preprocess`, `train`, `grid`, `predict`, `metrics`, `compile`, `infer`, `classify`.

**Not finished:**
- Phase 4 cleanup: `scripts/compile_run_metrics.py` and `compute_performance_metrics.py` are deprecation shims, but
  `evaluate_inference2.ipynb` and `save_final_prl_seg.ipynb` still import from `scripts.*`.
- Porting the "save final classifier-filtered segmentation" cells from `save_final_prl_seg.ipynb` into the CLI. This was an
  open goal in Pipeline_Notes, and `prl classify` may already cover part of it **(verify)**.
- The `training/bundles/prlsegresnet` bundle works through preprocess/postprocess, but nothing runs it end to end with AutoRunner yet,
  and it doesn't connect to `src/`.
- The August "Implementation Agnostic Plan" specs describe a full rewrite with manifests and provenance. No code exists for it yet.

**Gotchas noticed while reading:**
- In `evaluate_inference2.ipynb` cell 7, `experiment_name` is assigned twice, and the saved output shows
  `experiment.id == 'roi_train2/test_dicece_lambda/run1'`, **not** `stage6/run1`. Check which run today's re-run used.
- `save_final_prl_seg.ipynb` cell 9 loads `data` from `thalamus_project/data/working_data_wide3.csv`. That may be
  intentional (clinical variables), but double-check it.
- The `roi_train2`, `roi_train2_pmap` and `roi_train2_t1` training folders are near-copies.

---

## Open questions (carried over from your notes)

1. Why does classification on top of the T1 model perform much worse, when the T1 model's segmentation Dice is the same?
2. Should fold assignment be per subject? `stratified_folds.py` exists as a standalone experiment.
3. Does the LST-AI probability map with custom thresholding (instead of the binary mask) improve lesion and rim inputs?
4. How do subject-level PRL counts from the classified output compare with the counts in `full_inference_dataset`?
   `nPRL_diff` was started in `save_final_prl_seg.ipynb`.
5. Is MonaiLabel / 3D Slicer integration (the bundle) actually needed now, or later?

---

## Suggested restart (pick one, log it in FINDINGS_LOG.md)

1. **Make the classifier numbers trustworthy.** Swap in `StratifiedGroupKFold` by `subid` in `classify.py`, then report
   cross-validated AUC and the before/after Dice on held-out subjects only. This is small and self-contained, and it firms up your
   most presentable result.
2. **Count validation on the ~477-subject set.** Predicted PRL count vs. rater count per subject. This is the clinically meaningful
   test, and the pieces (`prl infer`, `prl classify`, `nPRL_diff`) mostly exist.
3. **Hold off on more refactoring** (finishing Phase 4, the bundle, the spec rewrite) until one of the two above is done.
   The current code is good enough to run them.