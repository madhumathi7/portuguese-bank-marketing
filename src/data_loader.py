"""Loading, integrity audit and the checks that must happen before modelling."""
from pathlib import Path
import numpy as np
import pandas as pd

from src import config


def find_data_file(data_dir=None):
    """Locate bank-additional-full.csv.

    The zip contains several variants. Your brief specifies
    Data > bank-additional > bank-additional-full.csv -- the 41,188-row file with
    all 20 inputs. bank-additional.csv is a 10% sample and bank-full.csv is the
    older 17-feature version without the macro indicators. Using the wrong one
    silently changes every number you report.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    exact = list(data_dir.rglob("bank-additional-full.csv"))
    if exact:
        return exact[0]
    others = list(data_dir.rglob("bank*.csv"))
    if others:
        raise FileNotFoundError(
            f"bank-additional-full.csv not found, but these exist: "
            f"{[p.name for p in others]}. Your brief requires the -full file.")
    raise FileNotFoundError(f"no bank*.csv under {data_dir}")


def load_raw(path=None):
    """Read the CSV. It is SEMICOLON separated, not comma.

    Loading it with the default comma separator produces a single-column frame
    with 41,188 rows and no error at all. It is the first thing that goes wrong
    with this dataset.
    """
    path = path or find_data_file()
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        raise ValueError("Parsed only one column. This file is ';' separated -- "
                         "pass sep=';' to read_csv.")
    return df, {"path": str(path), "rows": len(df), "columns": df.shape[1]}


def categorical_columns(df, exclude=()):
    """Text columns, in a way that works on both pandas 2 and pandas 3.

    PANDAS 3 TRAP: text columns now load as dtype 'str', not 'object'. So
    `df[c].dtype == object` is False and `select_dtypes(include="object")` raises
    a deprecation warning. Testing "not numeric" instead is stable across both
    versions and does not depend on which string backend is active.
    """
    return [c for c in df.columns
            if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]


def numeric_columns(df, exclude=()):
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def binarise_target(s):
    """'yes'/'no' -> 1/0, and pass through if it is already numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    return (s.astype("string").str.strip().str.lower() == "yes").astype(int)


def audit(df, target=config.TARGET):
    """Everything worth knowing before a single model is fitted."""
    cats = categorical_columns(df, exclude=(target,))
    nums = numeric_columns(df)

    unknown_counts = {c: int((df[c] == "unknown").sum()) for c in cats
                      if (df[c] == "unknown").any()}
    near_constant = {c: df[c].value_counts(normalize=True).iloc[0]
                     for c in df.columns
                     if df[c].value_counts(normalize=True).iloc[0] > 0.98}
    y = binarise_target(df[target])

    return {
        "rows": len(df), "columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        "categorical": cats, "numeric": nums,
        "true_nan_cells": int(df.isna().sum().sum()),
        "unknown_string_counts": unknown_counts,
        "duplicate_rows": int(df.duplicated().sum()),
        "near_constant_columns": {k: round(float(v), 4) for k, v in near_constant.items()},
        "positive_rate": round(float(y.mean()), 5),
        "imbalance_ratio": round(float((y == 0).sum() / max(y.sum(), 1)), 2),
        "accuracy_of_always_no": round(float(1 - y.mean()), 5),
        "pdays_sentinel_frac": (round(float((df["pdays"] == config.PDAYS_SENTINEL).mean()), 4)
                                if "pdays" in df else None),
    }


def duration_leak_evidence(df, target=config.TARGET):
    """Quantify why 'duration' must be dropped, rather than just asserting it.

    A reviewer should not have to take this on faith. Three pieces of evidence:
    duration alone achieves a near-perfect univariate AUC, the class means are
    far apart, and zero-duration calls are never successes. All three follow from
    the same fact -- the call ran long BECAUSE the customer was interested, so the
    feature is a consequence of the target, not a predictor of it.
    """
    from sklearn.metrics import roc_auc_score
    y = binarise_target(df[target])
    d = df["duration"]
    zero = d == 0
    return {
        "univariate_auc_of_duration": round(float(roc_auc_score(y, d)), 4),
        "mean_seconds_if_yes": round(float(d[y == 1].mean()), 1),
        "mean_seconds_if_no": round(float(d[y == 0].mean()), 1),
        "ratio": round(float(d[y == 1].mean() / max(d[y == 0].mean(), 1)), 2),
        "zero_duration_rows": int(zero.sum()),
        "zero_duration_success_rate": (round(float(y[zero].mean()), 4)
                                       if zero.any() else None),
    }


def split_features(df, drop_leaky=True, target=config.TARGET):
    """Return (X, y, feature_lists). drop_leaky=True is the production setting."""
    y = binarise_target(df[target]).to_numpy()
    X = df.drop(columns=[target])
    if drop_leaky:
        X = X.drop(columns=[c for c in config.LEAKY_FEATURES if c in X.columns])
    return X, y, {"categorical": categorical_columns(X),
                  "numeric": numeric_columns(X)}
