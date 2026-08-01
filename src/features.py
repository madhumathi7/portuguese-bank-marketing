"""Feature engineering and the preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from src import config
from src.data_loader import categorical_columns, numeric_columns


def fix_pdays(df):
    """Split the 999 sentinel into two honest features.

    pdays = 999 means "never previously contacted", not "contacted 999 days ago".
    Left as a raw number, 86% of the column sits at an extreme value that is not
    on the same scale as the rest: a linear model fits a slope through a cliff,
    and a tree wastes its first split rediscovering that 999 is special.

    The fix is to say what the data actually means: a binary flag for whether a
    previous contact happened, plus the real recency for the minority who were
    contacted. Filling the sentinel with a neutral value keeps the column dense
    while the flag carries the information.
    """
    out = df.copy()
    sent = out["pdays"] == config.PDAYS_SENTINEL
    out["was_contacted_before"] = (~sent).astype(int)
    real = out.loc[~sent, "pdays"]
    out["pdays_recency"] = out["pdays"].where(~sent, real.median() if len(real) else 0)
    return out.drop(columns=["pdays"])


def add_derived(df):
    """A handful of features the domain suggests. Each one is measurable, so
    drop any that does not earn its place rather than keeping it for tidiness."""
    out = df.copy()
    if "campaign" in out:
        # heavy right tail; the interesting distinction is 1-2 calls vs many
        out["campaign_capped"] = out["campaign"].clip(upper=10)
        out["is_first_contact"] = (out["campaign"] == 1).astype(int)
    if "age" in out:
        out["age_band"] = pd.cut(out["age"], [0, 25, 35, 45, 55, 65, 200],
                                 labels=["<25", "25-35", "35-45", "45-55",
                                         "55-65", "65+"]).astype("string")
    if "previous" in out:
        out["had_previous_campaign"] = (out["previous"] > 0).astype(int)
    return out


def engineer(df):
    """Full cleaning pass. Never mutates the input."""
    out = fix_pdays(df)
    out = add_derived(out)
    return out


def build_preprocessor(X, scale_numeric=True, education_ordinal=True):
    """ColumnTransformer: one-hot the categoricals, optionally scale the numerics.

    'unknown' is kept as a LEVEL, not treated as missing. It appears in six
    columns and is not a data-quality failure -- a customer declining to state
    their job or education is itself informative, and dropping or imputing those
    rows throws away roughly 12,800 records and whatever signal non-disclosure
    carries.

    handle_unknown='ignore' matters at serving time: a category the model never
    saw during training becomes all-zeros instead of raising. Without it the
    first unseen job title takes the service down.
    """
    cats = categorical_columns(X)
    nums = numeric_columns(X)

    ord_cols = []
    if education_ordinal and "education" in cats:
        ord_cols = ["education"]
        cats = [c for c in cats if c != "education"]

    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              min_frequency=10), cats),
        ("num", StandardScaler() if scale_numeric else "passthrough", nums),
    ]
    if ord_cols:
        # education has a real order; one-hot throws that away for no benefit
        transformers.append(
            ("ord", OrdinalEncoder(categories=[config.EDUCATION_ORDER + ["unknown"]],
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1), ord_cols))
    return ColumnTransformer(transformers, remainder="drop",
                             verbose_feature_names_out=False)


def feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return None
