"""Validation splits, including the time-aware one this dataset needs."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config


def verify_time_ordering(df, macro=None):
    """Check the evidence that the file is in chronological order.

    bank-additional-full.csv is documented as running May 2008 to November 2010
    in row order, but do not take that on trust -- test it. The macro indicators
    are calendar-driven, so if the file is time-ordered they should trend
    smoothly with the row index. A high absolute Spearman correlation between
    row number and euribor3m is the evidence.
    """
    macro = macro or [c for c in config.MACRO_FEATURES if c in df.columns]
    idx = np.arange(len(df))
    out = {}
    for c in macro:
        out[c] = round(float(pd.Series(idx).corr(df[c].reset_index(drop=True),
                                                 method="spearman")), 4)
    strongest = max(out.values(), key=abs) if out else 0.0
    return out, {
        "max_abs_spearman_with_row_index": round(abs(strongest), 4),
        "verdict": ("row order tracks calendar time -- a random split leaks the future"
                    if abs(strongest) > 0.5 else
                    "no strong time trend detected in row order; treat the temporal "
                    "split as indicative only"),
    }


def temporal_split(df, y, test_frac=0.2):
    """Train on the earlier rows, test on the later ones.

    WHY THIS MATTERS MORE THAN USUAL HERE. The macro indicators -- euribor3m,
    nr.employed, emp.var.rate -- move with the calendar, and this campaign ran
    straight through the 2008 financial crisis. Under a random split those
    columns let the model recognise WHEN a row comes from, and since rows from
    the same period share an outcome rate, that is a form of leakage: the model
    is partly reading the future.

    A random-CV score and a temporal-holdout score answer different questions.
    Random CV asks "how well does this generalise to more customers from the same
    period"; temporal asks "how well will it work next quarter", which is the one
    the bank is actually buying. Report both and explain the gap.
    """
    n = len(df)
    cut = int(n * (1 - test_frac))
    tr, te = np.arange(cut), np.arange(cut, n)
    return tr, te, {"train_rows": len(tr), "test_rows": len(te),
                    "train_positive_rate": round(float(np.mean(y[tr])), 4),
                    "test_positive_rate": round(float(np.mean(y[te])), 4)}


def stratified_folds(y, n_splits=config.N_SPLITS, seed=config.SEED):
    """Stratified is not optional at ~11% positives: plain KFold can hand a fold
    a materially different class balance, and the resulting spread is measurement
    noise rather than model variance."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
