import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from typing import TYPE_CHECKING

from prl.helpers import assign_prl_folds

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import BaseCrossValidator


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

class PrlKFold(BaseCrossValidator):

    def __init__(
        self,
        n_splits: int = 5,
        *,
        positive_label=1,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.positive_label = positive_label
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _assign_folds(self, y, groups) -> np.ndarray:
        if groups is None:
            raise ValueError(
                "The 'groups' parameter (subject IDs) must be provided."
            )
        if y is None:
            raise ValueError(
                "The 'y' parameter (target labels) must be provided."
            )

        meta = pd.DataFrame({"subid": np.asarray(groups), "target": np.asarray(y)})
        if meta.isna().any().any():
            raise ValueError(
                "Subject IDs and target labels must be non-missing."
            )

        subjects = (
            meta["target"]
            .eq(self.positive_label)
            .groupby(meta["subid"], observed=True, sort=False)
            .agg(n_prl="sum", n_lesions="size")
        )

        if not 2 <= self.n_splits <= len(subjects):
            raise ValueError(
                f"n_splits must be between 2 and {len(subjects)} subjects."
            )

        # Randomize ties reproducibly, then place largest PRL/lesion counts first
        subjects = subjects.sample(
            frac=1, random_state=self.random_state
        ).sort_values(["n_prl", "n_lesions"], ascending=False)

        # Greedy load balancing: each tuple tracks (PRLs, lesions, subjects)
        loads = [(0, 0, 0) for _ in range(self.n_splits)]
        subject_fold = {}

        for subject, row in subjects.iterrows():
            fold = min(range(self.n_splits), key=loads.__getitem__)
            n_prl, n_lesions, n_subjects = loads[fold]
            loads[fold] = (
                n_prl + row.n_prl,
                n_lesions + row.n_lesions,
                n_subjects + 1,
            )
            subject_fold[subject] = fold

        return meta["subid"].map(subject_fold).to_numpy(dtype=int)

    def split(self, X, y=None, groups=None):
        folds = self._assign_folds(y, groups)
        all_inds = np.arange(len(folds))

        for fold_idx in range(self.n_splits):
            test_inds = np.where(folds == fold_idx)[0]
            train_inds = np.setdiff1d(all_inds, test_inds, assume_unique=True)
            yield train_inds, test_inds


def assign_prl_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    *,
    subject_col: str = "subid",
    target_col: str = "case_type",
    positive_label: str = "PRL",
    random_state: int = 42,
) -> pd.Series:
    """Return a 1-based fold ID per row, balancing PRL counts across folds."""
    if df[[subject_col, target_col]].isna().any().any():
        raise ValueError("Subject IDs and case labels must be nonmissing.")

    subjects = (
        df[target_col].eq(positive_label)
        .groupby(df[subject_col], observed=True, sort=False)
        .agg(n_prl="sum", n_lesions="size")
    )
    if not 2 <= n_splits <= len(subjects):
        raise ValueError(f"n_splits must be between 2 and {len(subjects)}.")

    # Randomize ties reproducibly, then place the hardest subjects first.
    subjects = subjects.sample(frac=1, random_state=random_state).sort_values(
        ["n_prl", "n_lesions"], ascending=False
    )

    # Each tuple tracks (PRLs, lesions, subjects) in that fold.
    loads = [(0, 0, 0) for _ in range(n_splits)]
    subject_fold = {}

    for subject, row in subjects.iterrows():
        fold = min(range(n_splits), key=loads.__getitem__)
        n_prl, n_lesions, n_subjects = loads[fold]
        loads[fold] = (n_prl + row.n_prl, n_lesions + row.n_lesions, n_subjects + 1)
        subject_fold[subject] = fold + 1

    return df[subject_col].map(subject_fold).astype(int).rename("fold")




class RadiomicClassifier:
    def __init__(
        self,
        clf: BaseEstimator,
        data: pd.DataFrame,
        features: list[str]
    ):
        self.clf = clf
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        missing_features = [f for f in features if f not in data.columns]
        if any(missing_features):
            raise ValueError(f"Requested features missing: {features}")

        self.features = features
        self.data = data


    def _prepare_data(
        self,
        target: str = "case_type",
        pos_label: str = "PRL",
        neg_label: str = "Lesion",
        screen_col: str | None = "has_iron_infer",
        n_folds: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        
        data = self.data.rename(columns={target: "TARGET"})
        if screen_col is None:
            data["SCREEN"] = True
        else:
            data = data.rename(columns={screen_col: "SCREEN"})
            data["SCREEN"] = data["SCREEN"].fillna(False).astype(bool)

        data = self.data.loc[self.data[target].isin([pos_label, neg_label])].copy()
        data["fold"] = assign_prl_folds(data, n_splits=n_folds)

        """use in notebook for inspection
        # Inspect the achieved balance.
        summary = (
            data.assign(is_prl=data["TARGET"].eq(pos_label))
            .groupby("fold")
            .agg(
                subjects=("subid", "nunique"),
                lesions=("lesion_index", "size"),
                prls=("is_prl", "sum"),
                prl_prevalence=("is_prl", "mean"),
            )
        )
        print(summary.round(3))
        """

        assert data.groupby("subid")["fold"].nunique().eq(1).all()

        data = data.set_index(["subid", "lesion_index"])

        data["y_true"] = data[target].eq(POS_LABEL).astype(int)

        return data

    def get_cv_results(
        self,
        n_folds: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        
        data = self._prepare_data(n_folds, **kwargs)
        model = self.model
        FEATURES = self.features

        results_df = data[["TARGET", "SCREEN", "fold"]].copy()
        results_df["y_true"] = data["y_true"].copy()  # Move outside loop
        results_df["score"] = 0.0                     # Move outside loop

        for fold, test_df in data.groupby("fold"):
            # Stage 2 sees only screen-positive lesions from training subjects.
            X_train = data.loc[data["fold"].ne(fold) & data["SCREEN"]]
            model.fit(X_train[FEATURES], X_train["y_true"])

            X_test = test_df.loc[test_df["SCREEN"], FEATURES]
            if X_test.empty:
                continue

            if self.clf == "SVM":
                # Map SVC margins to positive ranking scores above screen rejects.
                # These scores are not calibrated probabilities.
                scores = expit(model.decision_function(X_test))
            else:
                scores = model.predict_proba(X_test)[:, 1]

            results_df.loc[X_test.index, "score"] = scores

        return results_df

# def prepare_data(df, **kwargs):
#     test_size = kwargs.get("test_size", TEST_SIZE)
#     cv_folds = kwargs.get("cv_folds", CV_FOLDS)
#     features = kwargs.get("features", FEATURES)
#     if features is None:
#         features = df.columns.drop(["subid", "lesion_index"])
#     df[features] = stats.zscore(df[features], nan_policy="omit")
#     X = df[features].values
#     try:
#         y = (df[TARGET] == POS_LABEL).astype(int).values   # 1 = PRL, 0 = Lesion
#     except KeyError:
#         y = np.full((X.shape[0],), np.nan)
    
#     if test_size == 1:
#         return None, X, None, y
#     elif test_size == 0:
#         return X, None, y, None

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
#     )
    
#     return X_train, X_test, y_train, y_test


# def train_model(X_train, y_train):
#     model = Pipeline([
#         ("imputer", SimpleImputer(strategy="constant", fill_value=0)),   # hull/sphere NaN when <4 rim voxels
#         ("scaler", StandardScaler()),
#         ("clf", LogisticRegression(
#             class_weight="balanced",    # important: 48 PRL vs ~117 Lesion
#             max_iter=1000,
#             random_state=RANDOM_STATE,
#         )),
#     ])
# #     model = Pipeline([
# #     ("imputer", SimpleImputer(strategy="constant", fill_value=0)),   
# #     ("scaler", StandardScaler()),
# #     ("clf", SVC(
# #         class_weight="balanced",     # Handles your 48 PRL vs ~117 Lesion imbalance
# #         random_state=RANDOM_STATE,
# #         probability=True             # Optional: Add this if you need predict_proba()
# #     )),
# # ])
#     model.fit(X_train, y_train)
#     return model


# def evaluate_model(model, X_test, y_test, display=False, to_print=False, return_fig=False, display_coef=False):
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]

#     if to_print:
#         print("\n--- Test set ---")
#         print(classification_report(y_test, y_pred, target_names=["Lesion", "PRL"]))
#         print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

#     if display:
#         fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#         ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Lesion", "PRL"]).plot(ax=axes[0])
#         axes[0].set_title("Confusion matrix (test set)")
#         RocCurveDisplay.from_predictions(y_test, y_prob, pos_label=1, ax=axes[1])
#         axes[1].set_title("ROC curve (test set)")
#         if return_fig:
#             return fig, axes
#         plt.tight_layout()
#         plt.show()
#     if display_coef:
#         fig, ax = plt.subplots(1, figsize=(6, 4))
#         coefs = model.named_steps["clf"].coef_[0]
#         coef_df = pd.DataFrame({"feature": FEATURES, "coef": coefs}).sort_values("coef", key=abs, ascending=False)
#         print("\n--- Feature coefficients (log-odds) ---")
#         print(coef_df.to_string(index=False))

#         ax.barh(coef_df["feature"], coef_df["coef"])
#         ax.axvline(0, color="black", linewidth=0.8)
#         ax.set_xlabel("Coefficient (log-odds)")
#         ax.set_title("Logistic regression coefficients")
#         if return_fig:
#             return fig, ax
#         plt.tight_layout()
#         plt.show()


#     return y_pred, y_prob 
    


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# FEATURE_LABELS = {
#     "rim_volume_infer": "Rim volume",
#     "rim_hull_volume_infer": "Convex hull volume",
#     "rim_sphere_radius_infer": "Enclosing sphere radius",
#     "pca_size": "PCA size",
#     "pca_sphericity": "PCA sphericity",
#     "pca_planarity": "PCA planarity",
#     "pca_linearity": "PCA linearity",
#     "mean_radius": "Mean radius",
#     "std_radius": "Radius SD",
#     "min_radius": "Minimum radius",
#     "max_radius": "Maximum radius",
#     "radial_cv": "Radial CV",
# }


# def plot_logistic_coefficients(
#     model,
#     features=FEATURES,
#     feature_labels=FEATURE_LABELS,
#     title="",
#     xlabel="Standardized logistic regression coefficient",
#     sort_by="coef",  # "coef" or "abs"
#     max_features=None,
#     figsize=(5.8, 4.2),
#     savepath=None,
#     ax=None,
# ):
#     """
#     Plot coefficients from a sklearn Pipeline containing:
#         imputer -> scaler -> LogisticRegression

#     Coefficients are interpretable as log-odds changes per 1 SD higher feature,
#     because the model includes StandardScaler before LogisticRegression.
#     """

#     if feature_labels is None:
#         feature_labels = {}

#     clf = model.named_steps["clf"]
#     coefs = clf.coef_[0]

#     coef_df = pd.DataFrame({
#         "feature": features,
#         "coef": coefs,
#     })

#     coef_df["label"] = coef_df["feature"].map(feature_labels).fillna(coef_df["feature"])

#     if sort_by == "abs":
#         coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values().index)
#     elif sort_by == "coef":
#         coef_df = coef_df.sort_values("coef")
#     else:
#         raise ValueError("sort_by must be 'coef' or 'abs'")

#     if max_features is not None:
#         if sort_by == "abs":
#             coef_df = coef_df.tail(max_features)
#         else:
#             # keep largest magnitude features, then sort by coefficient for display
#             coef_df = (
#                 coef_df.assign(abs_coef=coef_df["coef"].abs())
#                 .sort_values("abs_coef")
#                 .tail(max_features)
#                 .sort_values("coef")
#                 .drop(columns="abs_coef")
#             )

#     if ax is None:
#         fig, ax = plt.subplots(figsize=figsize)
#     else:
#         fig = ax.figure

#     y = np.arange(len(coef_df))

#     # Optional color split by sign. If you want all one color, replace with a single string.
#     colors = np.where(coef_df["coef"] >= 0, "#2F4F4F", "#8B3A3A")

#     ax.barh(
#         y,
#         coef_df["coef"],
#         color=colors,
#         alpha=0.9,
#         height=0.72,
#     )

#     ax.axvline(0, color="0.15", linewidth=1.1)

#     ax.set_yticks(y)
#     ax.set_yticklabels(coef_df["label"], fontsize=10)
#     ax.set_xlabel(xlabel, fontsize=10.5)
#     ax.set_title(title, fontsize=12.5, weight="bold", pad=8)

#     # Add directional annotations
#     x_min, x_max = ax.get_xlim()
#     x_range = x_max - x_min

#     # ax.text(
#     #     x_max,
#     #     len(coef_df) - 0.15,
#     #     "More likely PRL",
#     #     ha="right",
#     #     va="bottom",
#     #     fontsize=8.5,
#     #     color="0.25",
#     # )

#     # ax.text(
#     #     x_min,
#     #     len(coef_df) - 0.15,
#     #     "More likely non-PRL",
#     #     ha="left",
#     #     va="bottom",
#     #     fontsize=8.5,
#     #     color="0.25",
#     # )

#     # Clean style
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.grid(axis="x", color="0.9", linewidth=0.8)
#     ax.tick_params(axis="x", labelsize=9.5)
#     ax.tick_params(axis="y", length=0)

#     # Make x-axis symmetric-ish if helpful
#     max_abs = np.nanmax(np.abs(coef_df["coef"]))
#     ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)

#     fig.tight_layout()

#     if savepath is not None:
#         fig.savefig(savepath, dpi=600, bbox_inches="tight", transparent=True)

#     return fig, ax, coef_df


# def cross_validation():
#     pass

