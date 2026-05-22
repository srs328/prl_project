import numpy as np
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
    try:
        y = (df[TARGET] == POS_LABEL).astype(int).values   # 1 = PRL, 0 = Lesion
    except KeyError:
        y = np.full((X.shape[0],), np.nan)
    
    if test_size == 1:
        return None, X, None, y
    elif test_size == 0:
        return X, None, y, None

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


def evaluate_model(model, X_test, y_test, display=False, to_print=False, return_fig=False, display_coef=False):
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
        if return_fig:
            return fig, axes
        plt.tight_layout()
        plt.show()
    if display_coef:
        fig, ax = plt.subplots(1, figsize=(6, 4))
        coefs = model.named_steps["clf"].coef_[0]
        coef_df = pd.DataFrame({"feature": FEATURES, "coef": coefs}).sort_values("coef", key=abs, ascending=False)
        print("\n--- Feature coefficients (log-odds) ---")
        print(coef_df.to_string(index=False))

        ax.barh(coef_df["feature"], coef_df["coef"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient (log-odds)")
        ax.set_title("Logistic regression coefficients")
        if return_fig:
            return fig, ax
        plt.tight_layout()
        plt.show()


    return y_pred, y_prob 
    


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_LABELS = {
    "rim_volume_infer": "Rim volume",
    "rim_hull_volume_infer": "Convex hull volume",
    "rim_sphere_radius_infer": "Enclosing sphere radius",
    "pca_size": "PCA size",
    "pca_sphericity": "PCA sphericity",
    "pca_planarity": "PCA planarity",
    "pca_linearity": "PCA linearity",
    "mean_radius": "Mean radius",
    "std_radius": "Radius SD",
    "min_radius": "Minimum radius",
    "max_radius": "Maximum radius",
    "radial_cv": "Radial CV",
}


def plot_logistic_coefficients(
    model,
    features=FEATURES,
    feature_labels=FEATURE_LABELS,
    title="",
    xlabel="Standardized logistic regression coefficient",
    sort_by="coef",  # "coef" or "abs"
    max_features=None,
    figsize=(5.8, 4.2),
    savepath=None,
    ax=None,
):
    """
    Plot coefficients from a sklearn Pipeline containing:
        imputer -> scaler -> LogisticRegression

    Coefficients are interpretable as log-odds changes per 1 SD higher feature,
    because the model includes StandardScaler before LogisticRegression.
    """

    if feature_labels is None:
        feature_labels = {}

    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]

    coef_df = pd.DataFrame({
        "feature": features,
        "coef": coefs,
    })

    coef_df["label"] = coef_df["feature"].map(feature_labels).fillna(coef_df["feature"])

    if sort_by == "abs":
        coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values().index)
    elif sort_by == "coef":
        coef_df = coef_df.sort_values("coef")
    else:
        raise ValueError("sort_by must be 'coef' or 'abs'")

    if max_features is not None:
        if sort_by == "abs":
            coef_df = coef_df.tail(max_features)
        else:
            # keep largest magnitude features, then sort by coefficient for display
            coef_df = (
                coef_df.assign(abs_coef=coef_df["coef"].abs())
                .sort_values("abs_coef")
                .tail(max_features)
                .sort_values("coef")
                .drop(columns="abs_coef")
            )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y = np.arange(len(coef_df))

    # Optional color split by sign. If you want all one color, replace with a single string.
    colors = np.where(coef_df["coef"] >= 0, "#2F4F4F", "#8B3A3A")

    ax.barh(
        y,
        coef_df["coef"],
        color=colors,
        alpha=0.9,
        height=0.72,
    )

    ax.axvline(0, color="0.15", linewidth=1.1)

    ax.set_yticks(y)
    ax.set_yticklabels(coef_df["label"], fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_title(title, fontsize=12.5, weight="bold", pad=8)

    # Add directional annotations
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min

    # ax.text(
    #     x_max,
    #     len(coef_df) - 0.15,
    #     "More likely PRL",
    #     ha="right",
    #     va="bottom",
    #     fontsize=8.5,
    #     color="0.25",
    # )

    # ax.text(
    #     x_min,
    #     len(coef_df) - 0.15,
    #     "More likely non-PRL",
    #     ha="left",
    #     va="bottom",
    #     fontsize=8.5,
    #     color="0.25",
    # )

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", length=0)

    # Make x-axis symmetric-ish if helpful
    max_abs = np.nanmax(np.abs(coef_df["coef"]))
    ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=600, bbox_inches="tight", transparent=True)

    return fig, ax, coef_df


def cross_validation():
    pass

