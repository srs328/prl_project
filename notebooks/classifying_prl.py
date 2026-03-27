# %%
"""
Quick PRL classifier skeleton.
Run top-to-bottom. Port to notebook as needed.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# %%
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV = "/home/srs-9/Projects/prl_project/analysis/prl_image_stats-roi_train2_stage3_numcrops_bkd_constwt115_run2.csv"

# Features to use — extend this list when radiomics arrive
FEATURES = [
    "rim_volume_infer",
    "rim_hull_volume_infer",
    "rim_sphere_radius_infer",
    "lesion_volume_infer",
    "lesion_hull_volume_infer",
]

TARGET = "case_type"        # "PRL" or "Lesion"
POS_LABEL = "PRL"

TEST_SIZE = 0.25            # fraction held out
RANDOM_STATE = 42
CV_FOLDS = 5

# %%
# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV)
df = df[df['has_iron_infer']]
print(f"Loaded {len(df)} rows | {df[TARGET].value_counts().to_dict()}")

X = df[FEATURES].values
y = (df[TARGET] == POS_LABEL).astype(int).values   # 1 = PRL, 0 = Lesion

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {len(y_train)} ({y_train.sum()} PRL) | Test: {len(y_test)} ({y_test.sum()} PRL)")


# %%
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),   # hull/sphere NaN when <4 rim voxels
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",    # important: 48 PRL vs ~117 Lesion
        max_iter=1000,
        random_state=RANDOM_STATE,
    )),
])

model.fit(X_train, y_train)

# %%
# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Test set ---")
print(classification_report(y_test, y_pred, target_names=["Lesion", "PRL"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Lesion", "PRL"]).plot(ax=axes[0])
axes[0].set_title("Confusion matrix (test set)")
RocCurveDisplay.from_predictions(y_test, y_prob, pos_label=1, ax=axes[1])
axes[1].set_title("ROC curve (test set)")
plt.tight_layout()
plt.show()

# %%
# ---------------------------------------------------------------------------
# Cross-validation (better estimate given small N)
# ---------------------------------------------------------------------------

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(
    model, X, y, cv=cv,
    scoring=["roc_auc", "f1", "precision", "recall"],
    return_train_score=False,
)

print(f"\n--- {CV_FOLDS}-fold stratified CV ---")
for metric, vals in cv_results.items():
    if metric.startswith("test_"):
        name = metric.replace("test_", "")
        print(f"  {name:12s}: {vals.mean():.3f} ± {vals.std():.3f}")

# %%
# ---------------------------------------------------------------------------
# Feature coefficients
# ---------------------------------------------------------------------------

coefs = model.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature": FEATURES, "coef": coefs}).sort_values("coef", key=abs, ascending=False)
print("\n--- Feature coefficients (log-odds) ---")
print(coef_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(coef_df["feature"], coef_df["coef"])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (log-odds)")
ax.set_title("Logistic regression coefficients")
plt.tight_layout()
plt.show()
