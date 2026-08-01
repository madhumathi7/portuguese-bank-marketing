"""The model zoo. Each entry tests a stated hypothesis."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

from src import config
from src.features import build_preprocessor


def build_models(X, scale_pos_weight=1.0, seed=config.SEED, n_jobs=-1):
    """Six candidates wrapped with preprocessing so nothing leaks across folds.

    LogisticRegression  Interpretable coefficients, which matters because Task 3
                        asks for advice a marketing team can act on. An odds
                        ratio is something you can put in a slide; a boosted
                        ensemble's 400th tree is not.
    DecisionTree        Deliberately shallow. Produces readable rules -- "if
                        poutcome is success and contact is cellular, then..." --
                        that translate straight into campaign policy.
    RandomForest        Bagged trees, robust to the mixed types here, a solid
                        non-linear reference.
    LightGBM            Usually the strongest on tabular data of this shape and
                        size, and fast enough to retrain nightly.
    XGBoost             Second boosting implementation as a cross-check that a
                        result is not an artefact of one library's defaults.
    LightGBM_balanced   Identical but with scale_pos_weight, to isolate the
                        effect of imbalance handling from everything else.

    Every model is wrapped in a Pipeline with the preprocessor INSIDE. That is
    what makes cross-validation honest: the encoder and scaler are refitted on
    each training fold rather than on the whole dataset.
    """
    def pipe(clf, scale=True):
        return Pipeline([("prep", build_preprocessor(X, scale_numeric=scale)),
                         ("clf", clf)])

    return {
        "LogisticRegression": pipe(LogisticRegression(
            C=0.5, max_iter=3000, solver="lbfgs", random_state=seed)),
        "DecisionTree": pipe(DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=200, class_weight="balanced",
            random_state=seed), scale=False),
        "RandomForest": pipe(RandomForestClassifier(
            n_estimators=400, max_depth=14, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=n_jobs,
            random_state=seed), scale=False),
        "LightGBM": pipe(lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, max_depth=6,
            min_child_samples=40, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=n_jobs,
            random_state=seed, verbose=-1), scale=False),
        "XGBoost": pipe(xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=5,
            min_child_weight=10, subsample=0.85, colsample_bytree=0.8,
            reg_lambda=1.0, eval_metric="auc", n_jobs=n_jobs,
            random_state=seed, tree_method="hist"), scale=False),
        "LightGBM_balanced": pipe(lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, max_depth=6,
            min_child_samples=40, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, n_jobs=n_jobs,
            random_state=seed, verbose=-1), scale=False),
    }


def imbalance_ratio(y):
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))
