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


TEST_SIZE = 0.25            # fraction held out
RANDOM_STATE = 42
CV_FOLDS = 5
TARGET = "case_type"        # "PRL" or "Lesion"
POS_LABEL = "PRL"


FEATURES = [
    "rim_volume_infer",
    "rim_hull_volume_infer",
    "rim_sphere_radius_infer",
    "pca_size",
    "pca_sphericity",

    "pca_planarity",
    "pca_linearity",
    "mean_radius",
    "std_radius",
    "min_radius",
    "max_radius",
    "radial_cv",
]



def prepare_data(df, **kwargs):
    test_size = kwargs.get("test_size", TEST_SIZE)
    cv_folds = kwargs.get("cv_folds", CV_FOLDS)
    features = kwargs.get("features", FEATURES)
    X = df[features].values
    y = (df[TARGET] == POS_LABEL).astype(int).values   # 1 = PRL, 0 = Lesion

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
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
    return model


def evaluate_model(model, X_test, y_test, display=False, to_print=False):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    if to_print:
        print("\n--- Test set ---")
        print(classification_report(y_test, y_pred, target_names=["Lesion", "PRL"]))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    if display:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Lesion", "PRL"]).plot(ax=axes[0])
        axes[0].set_title("Confusion matrix (test set)")
        RocCurveDisplay.from_predictions(y_test, y_prob, pos_label=1, ax=axes[1])
        axes[1].set_title("ROC curve (test set)")
        plt.tight_layout()
        plt.show()


def cross_validation():
    pass

