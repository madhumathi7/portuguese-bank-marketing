"""Tests for the traps this dataset sets."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import (categorical_columns, numeric_columns, binarise_target,
                             duration_leak_evidence, split_features)
from src.features import fix_pdays, engineer
from src.train import precision_at_k, lift_at_k, profit_curve, choose_threshold


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(20, 70, n),
        "job": rng.choice(["admin.", "blue-collar", "unknown"], n),
        "education": rng.choice(["basic.4y", "university.degree", "unknown"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "month": rng.choice(["may", "oct"], n),
        "day_of_week": rng.choice(["mon", "fri"], n),
        "campaign": rng.integers(1, 8, n),
        "previous": rng.integers(0, 3, n),
        "pdays": rng.choice([999, 3, 6, 12], n, p=[.85, .05, .05, .05]),
        "poutcome": rng.choice(["nonexistent", "failure", "success"], n),
        "duration": rng.integers(0, 900, n),
        "y": rng.choice(["no", "yes"], n, p=[.89, .11]),
    })


def test_target_binarises_from_yes_no():
    s = pd.Series(["yes", "no", "YES", " no "])
    assert binarise_target(s).tolist() == [1, 0, 1, 0]


def test_column_typing_survives_pandas3_string_dtype():
    """pandas 3 loads text as dtype 'str', not 'object', so `== object` fails.
    Testing 'not numeric' instead works on both pandas 2 and 3."""
    d = _frame()
    cats = categorical_columns(d, exclude=(config.TARGET,))
    assert "job" in cats and "age" not in cats
    assert "age" in numeric_columns(d) and "job" not in numeric_columns(d)


def test_pdays_sentinel_is_split_not_treated_as_a_number():
    d = _frame()
    out = fix_pdays(d)
    assert "pdays" not in out.columns, "raw pdays must not survive"
    assert set(out["was_contacted_before"].unique()) <= {0, 1}
    assert out["pdays_recency"].max() < config.PDAYS_SENTINEL, \
        "999 must never reach the model as a magnitude"


def test_duration_is_dropped_from_production_features():
    d = _frame()
    X, y, _ = split_features(engineer(d), drop_leaky=True)
    assert "duration" not in X.columns
    X2, _, _ = split_features(engineer(d), drop_leaky=False)
    assert "duration" in X2.columns, "benchmark mode must keep it"


def test_duration_leak_evidence_reports_separation():
    d = _frame()
    d.loc[d.y == "yes", "duration"] = 900          # make the leak explicit
    ev = duration_leak_evidence(d)
    assert ev["univariate_auc_of_duration"] > 0.8
    assert ev["ratio"] > 1.5


def test_unknown_is_kept_as_a_level_not_dropped():
    d = _frame()
    X, _, cols = split_features(engineer(d))
    assert "unknown" in set(X["job"].unique()), "'unknown' is information, not missingness"


def test_precision_at_k_and_lift():
    y = np.array([0] * 90 + [1] * 10)
    perfect = np.linspace(0, 1, 100)
    assert precision_at_k(y, perfect, 0.10) == 1.0
    assert lift_at_k(y, perfect, 0.10) == pytest.approx(10.0)


def test_profit_curve_prefers_a_subset_when_calls_cost_more_than_they_earn():
    """If value x base_rate < cost, calling everyone must lose money."""
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.10).astype(int)
    prob = y * 0.5 + rng.random(2000) * 0.5           # informative but noisy
    curve, best = profit_curve(y, prob, cost_per_call=8.0, value_per_conversion=60.0)
    assert best["profit_calling_everyone"] < best["best_profit"]
    assert 0 < best["best_contacted_frac"] < 1.0


def test_threshold_is_below_the_naive_half():
    rng = np.random.default_rng(1)
    prob = rng.beta(2, 16, 4000)
    t = choose_threshold(prob, 0.10)
    assert 0 < t < 0.5
    assert abs((prob >= t).mean() - 0.10) < 0.02
