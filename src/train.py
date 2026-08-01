"""Cross-validation, temporal evaluation, thresholds and campaign economics."""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, brier_score_loss)

from src import config
from src.split import stratified_folds


def precision_at_k(y_true, prob, frac=0.10):
    """Of the top k% by score, what fraction actually subscribe?

    The call centre has finite capacity, so this is the number the marketing team
    lives with. F1 optimises a threshold nobody chose.
    """
    k = max(1, int(len(prob) * frac))
    idx = np.argsort(-np.asarray(prob))[:k]
    return float(np.asarray(y_true)[idx].mean())


def lift_at_k(y_true, prob, frac=0.10):
    """How many times better than calling a random k% of the list."""
    base = float(np.mean(y_true))
    return precision_at_k(y_true, prob, frac) / base if base else np.nan


def cross_validate(model_fn, X, y, model_name, n_splits=config.N_SPLITS,
                   seed=config.SEED, threshold=0.5, verbose=True):
    """Stratified K-fold with the preprocessor refitted inside every fold."""
    y = np.asarray(y)
    skf = stratified_folds(y, n_splits, seed)
    oof = np.zeros(len(X), dtype="float64")
    rows = []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = model_fn()
        t0 = time.perf_counter()
        model.fit(X.iloc[tr], y[tr])
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        prob = model.predict_proba(X.iloc[va])[:, 1]
        pred_s = time.perf_counter() - t1
        oof[va] = prob

        pred = (prob >= threshold).astype(int)
        rows.append({"model": model_name, "fold": fold,
                     "roc_auc": roc_auc_score(y[va], prob),
                     "pr_auc": average_precision_score(y[va], prob),
                     "precision_at_10": precision_at_k(y[va], prob),
                     "lift_at_10": lift_at_k(y[va], prob),
                     "brier": brier_score_loss(y[va], prob),
                     "precision": precision_score(y[va], pred, zero_division=0),
                     "recall": recall_score(y[va], pred, zero_division=0),
                     "f1": f1_score(y[va], pred, zero_division=0),
                     "fit_seconds": fit_s,
                     "predict_ms_per_1k": pred_s / len(va) * 1e6})
        if verbose:
            r = rows[-1]
            print(f"  fold {fold}  AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.4f}"
                  f"  lift@10% {r['lift_at_10']:.2f}x  ({fit_s:.1f}s)")

    df = pd.DataFrame(rows)
    summary = {"model": model_name,
               "oof_roc_auc": float(roc_auc_score(y, oof)),
               "oof_pr_auc": float(average_precision_score(y, oof)),
               "oof_precision_at_10": precision_at_k(y, oof),
               "oof_lift_at_10": lift_at_k(y, oof),
               "oof_brier": float(brier_score_loss(y, oof)),
               "roc_auc_std": float(df["roc_auc"].std()),
               **{f"{m}_mean": float(df[m].mean()) for m in
                  ["precision", "recall", "f1"]},
               "fit_seconds_total": float(df["fit_seconds"].sum()),
               "predict_ms_per_1k": float(df["predict_ms_per_1k"].mean())}
    return df, summary, oof


def temporal_evaluate(model_fn, X, y, tr_idx, te_idx, model_name):
    """Fit on the earlier period, score the later one.

    Expect this number to be LOWER than the cross-validated one. That gap is not
    a bug and it is not something to hide -- it is the cost of the macro
    indicators encoding time, and it is the more realistic estimate of what the
    bank will see next quarter.
    """
    y = np.asarray(y)
    model = model_fn()
    model.fit(X.iloc[tr_idx], y[tr_idx])
    prob = model.predict_proba(X.iloc[te_idx])[:, 1]
    yt = y[te_idx]
    return {"model": model_name,
            "temporal_roc_auc": float(roc_auc_score(yt, prob)),
            "temporal_pr_auc": float(average_precision_score(yt, prob)),
            "temporal_lift_at_10": lift_at_k(yt, prob)}, prob


def profit_curve(y_true, prob, cost_per_call=config.COST_PER_CALL,
                 value_per_conversion=config.VALUE_PER_CONVERSION, points=100):
    """Expected campaign profit as a function of how many people you call.

    Cheap to compute and far more persuasive than an F1 score. It converts the
    model into the only currency a marketing director cares about, and it shows
    the optimum is usually NOT "call everybody" and usually NOT "call the top 1%".
    Replace the two constants with the bank's real figures before quoting any
    number from this.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(prob))
    ys = y_true[order]
    fracs = np.linspace(0.01, 1.0, points)
    rows = []
    for f in fracs:
        k = max(1, int(len(ys) * f))
        conv = ys[:k].sum()
        profit = conv * value_per_conversion - k * cost_per_call
        rows.append({"contacted_frac": f, "contacted": k, "conversions": int(conv),
                     "precision": conv / k, "recall": conv / max(ys.sum(), 1),
                     "profit": profit})
    out = pd.DataFrame(rows)
    best = out.loc[out["profit"].idxmax()]
    return out, {"best_contacted_frac": float(best["contacted_frac"]),
                 "best_profit": float(best["profit"]),
                 "profit_calling_everyone": float(out.iloc[-1]["profit"]),
                 "precision_at_optimum": float(best["precision"]),
                 "recall_at_optimum": float(best["recall"])}


def choose_threshold(prob, frac):
    """Score at the chosen capacity. 0.5 is a default, not a decision -- under
    11% positives a calibrated model rarely scores anyone above it."""
    return float(np.quantile(np.asarray(prob), 1 - frac))
