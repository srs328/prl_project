# PRL Project — Findings Log

Produced by [Claude](https://claude.ai/chat/18345581-241a-421a-8bae-37b9add08b22) on 11/13/2026

What I learned and what I decided, newest entry first. This is the research record, not the code plan
(code plans go in `plan.md`, `notes/Next_refactor_step.md`, etc.). 

**Rules for myself**
- One entry per question answered, or per decision made. Short is fine.
- Always record *which run / which data* produced the result (run ID + commit hash) so I can trace it later.
- Write the entry the same day. If it's not written down, it didn't happen.
- Never edit old entries. If something turns out wrong, add a new entry that says so and links back.
- Separate "what the data showed" from "what I think it means".

---

## Entry template

```markdown
### YYYY-MM-DD — <short title phrased as the question or decision>

**Question:** What was I trying to find out?
**Runs / data:** run IDs (e.g. `roi_train2/stage6_sweep_dicece_wts/run1`), dataset, commit `abc1234`,
notebook or script used, output file path(s)
**Result:** What the numbers or images actually showed. Include n (cases / subjects) and spread, not just a mean.
**Interpretation:** What I think it means. Note how confident I am and what could make it wrong.
**Decision / next step:** What I'm going to do because of this (or "none, just noting").
**Tags:** #segmentation #classifier #data #bug #infra #decision
```

---

## Entries

> The entries below were reconstructed in Nov 2026 from notes, notebooks, CSVs, and commit history.
> Dates are approximate (commit dates). Anything marked **(verify)** needs a check against the actual files on disk / HPC.

### 2026-11-13 — Restarting after ~3 months away; findings log started

**Question:** Where did I leave off?
**Runs / data:** see `PROJECT_STATUS_2026-11.md`
**Result:** Last substantive analysis work was May–June 2026. In August I wrote implementation-agnostic pipeline specs
(`notes/Implementation Agnostic Plan/`). Nothing was trained after that.
**Interpretation:** —
**Decision / next step:** Pick one thread to resume (see the status doc's "Suggested restart" section) before touching any refactor.
**Tags:** #decision

### 2026-05-22 — Lesion-level classifier on top of SegResNet output improves rim Dice modestly

**Question:** Does filtering SegResNet's predicted rims with a lesion-level logistic regression (shape + radiomic features) improve rim segmentation?
**Runs / data:** Segmentation from `roi_train2/stage6_sweep_dicece_wts/run1`. Classifier = `src/analysis/image/classify.py`
(logistic regression). Features: rim volume, hull volume, PCA shape metrics, radius stats, pyradiomics GLCM features.
Notebook `notebooks/save_final_prl_seg.ipynb`.
**Result:**
- Classifier held-out test: ROC-AUC ~0.86–0.88; PRL class precision ~0.67–0.75, recall ~0.77–0.80 (n ≈ 125–139 lesions, ~40 PRL).
- Full-brain rim performance on the 24 training subjects, before → after classification: Dice 0.600 → 0.632,
  precision 0.50 → 0.57, recall 0.77 → 0.74, HD95 35.1 → 14.6.
**Interpretation:** The classifier mainly removes false-positive iron (precision up, HD95 much better) at a small cost to recall.
**Caveats:** (1) The model used for the before/after comparison was refit with `test_size=0` on data that includes these
subjects. The notebook comment acknowledges it's "double dipping", so the after-classification numbers are optimistic.
(2) `prepare_data` splits at the **lesion level**, stratified by label but not grouped by subject, so lesions from the same
subject can land in both train and test. Treat the AUC as optimistic until it's redone with subject-grouped CV.
**Decision / next step:** Redo the classifier evaluation with `GroupKFold` / `StratifiedGroupKFold` by subject. **(verify)** whether the Dice comparison used all 24 subjects or test subjects only.
**Tags:** #classifier #caveat

### 2026-05 — T1 as a third channel: similar segmentation Dice, much worse downstream classification

**Question:** Does adding T1 (flair+phase+t1) help?
**Runs / data:** `roi_train2_t1/sweep_dicecewts/run1`, `roi_train2_t1/test_dicece_lambda/run1` vs `roi_train2` equivalents.
**Result:** Rim Dice on PRL test cases was about the same (0.587 / 0.586 vs 0.58–0.60 for two-channel runs). Per Pipeline_Notes, classification built on the T1 run's segmentations performed "much worse" than classification built on `stage6/run1`.
**Interpretation:** Unexplained. Open question in Pipeline_Notes.
**Decision / next step:** Open. One idea: compare the per-lesion feature distributions (rim volume, has_iron counts) between the two runs' predictions before blaming the classifier.
**Tags:** #segmentation #classifier #open

### 2026-03/05 — DiceCE class weighting was the single biggest win; everything after is within noise

**Question:** Which training settings matter for rim Dice?
**Runs / data:** stages 1–8 under `roi_train2/` plus `test_dicece_lambda`. Summary table at `notebooks/hpo_performance_metrics.csv`.
Metric = mean rim Dice over PRL test cases (**only 9 PRL test cases**).
**Result:**
- Baseline runs (`run1`–`run3`): 0.41–0.55.
- Stage 1 (crop_ratios × lr): 0.50–0.55; crop_ratios oversampling didn't help.
- Stage 2: adding DiceCE weight `[1,1,5]` jumped 0.52 → 0.59. include_background made little difference.
- Stages 3–8: 0.52–0.61. Best: `stage8_push_epochs/run1` 0.613, `stage6/run1` ([1,1,10]) 0.604, `stage6/run3` ([1,1,15]) 0.603.
  Raising the lesion weight (middle term >1) consistently *hurt* (stage 4/5).
- Per-run std is ~0.10–0.13, so differences of 0.01–0.02 between the top runs aren't meaningful with n = 9.
**Interpretation:** Rim weighting matters a lot. The rest is tuning noise at this sample size.
**Decision / next step:** `stage6_sweep_dicece_wts/run1` chosen as the working model for classification.
**Tags:** #segmentation #decision

### 2026-03 — Bug: num_crops_per_image was silently 2 in every early run

**Question:** Were the sweep settings actually applied?
**Runs / data:** roughly run5 through stage 2. Details in `notes/llm_question_pad.md`.
**Result:** SegResNet's `auto_adjust_network_settings` overrides `batch_size` unless `auto_scale_batch: false`. Then in
`crop_mode == "ratio"` it sets `num_crops_per_image = batch_size; batch_size = 1`. Effectively every run had num_crops = 2, batch = 1.
Also: when num_crops_per_image > 1, SegResNet scales `num_epochs` down (e.g. 500 → 166 at 4 crops), so total training differs.
**Interpretation:** The settings I passed in are not the settings that were used. Runs with different num_crops aren't equal-compute comparisons.
**Decision / next step:** From stage 5 onward, set `auto_scale_allowed: false` and `auto_scale_batch: false`. Always check the
*resolved* `hyper_parameters.yaml` in the run directory, not the input config.
**Tags:** #bug #infra

### 2026-03-20 — Fold 4 underperformed because of larger lesions

**Question:** Why is fold 4's validation worse?
**Runs / data:** `roi_train2/run2`; `notes/CHAT_SESSION_SUMMARY.md`.
**Result:** Fold 4 crops averaged ~40% larger than other folds.
**Interpretation:** Distribution shift from fold assignment, not label quality.
**Decision / next step:** Stratify folds (lesion size; later, per-subject assignment). A standalone experiment exists at
`training/bundles/prlsegresnet/experiments/stratified_folds.py`. **(verify)** whether it was ever applied to a training dataset.
**Tags:** #data #decision