"""Every figure. Saves to reports/figures and returns the figure."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve

from src import config

INDIGO, AMBER, ROSE, SLATE = "#4A3B6B", "#7A6320", "#9C4A52", "#556270"
PALETTE = [INDIGO, AMBER, ROSE, SLATE, "#2F6B5E", "#8A5A2B"]
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    if name:
        fig.savefig(config.FIGURES_DIR / name, dpi=140, bbox_inches="tight")
    return fig


def target_balance(y, name="target_balance.png"):
    y = pd.Series(np.asarray(y))
    c = y.value_counts().sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(["no (0)", "yes (1)"], c.values, color=[SLATE, INDIGO], width=.55)
    for i, v in enumerate(c.values):
        ax[0].text(i, v, f"{v:,}\n{v/len(y):.1%}", ha="center", va="bottom", fontsize=10)
    ax[0].set(title="term deposit subscriptions", ylabel="customers"); ax[0].margins(y=.18)
    ax[1].barh(["'always no' accuracy", "actual yes rate"], [1-y.mean(), y.mean()],
               color=[SLATE, INDIGO], height=.45)
    ax[1].set(xlim=(0, 1), title="why accuracy must not be the headline")
    ax[1].text(1-y.mean(), 0, f"  {1-y.mean():.1%} with zero skill", va="center", fontsize=10)
    fig.tight_layout(); return _save(fig, name)


def duration_leak(df, evidence, target=config.TARGET, name="duration_leak.png"):
    y = (df[target].astype("string").str.lower() == "yes") if not pd.api.types.is_numeric_dtype(df[target]) else df[target] == 1
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for flag, col, lab in ((False, SLATE, "no"), (True, ROSE, "yes")):
        sns.kdeplot(df.loc[y == flag, "duration"].clip(upper=1500), ax=ax[0],
                    color=col, fill=True, alpha=.3, label=f"y = {lab}")
    ax[0].set(xlabel="call duration (seconds)",
              title=f"duration alone scores AUC {evidence['univariate_auc_of_duration']:.3f}")
    ax[0].legend()
    ax[1].bar(["y = no", "y = yes"], [evidence["mean_seconds_if_no"], evidence["mean_seconds_if_yes"]],
              color=[SLATE, ROSE], width=.5)
    ax[1].set(ylabel="mean seconds",
              title="the call ran long BECAUSE they said yes\n(a consequence, not a predictor)")
    fig.tight_layout(); return _save(fig, name)


def leak_gap(with_auc, without_auc, name="leak_gap.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(["with duration\n(benchmark only)", "without duration\n(production)"],
                   [with_auc, without_auc], color=[ROSE, INDIGO], height=.5)
    for b, v in zip(bars, [with_auc, without_auc]):
        ax.text(v + .005, b.get_y() + b.get_height()/2, f"{v:.4f}", va="center", fontsize=11)
    ax.set(xlim=(0.5, 1.0), xlabel="ROC-AUC",
           title=f"the leakage gap: {with_auc-without_auc:+.3f} AUC of pure hindsight")
    return _save(fig, name)


def pdays_sentinel(df, name="pdays_sentinel.png"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    ax[0].hist(df["pdays"], bins=60, color=SLATE)
    ax[0].set(xlabel="pdays as loaded", title="86% of the column sits at 999")
    real = df.loc[df["pdays"] != config.PDAYS_SENTINEL, "pdays"]
    ax[1].hist(real, bins=30, color=INDIGO)
    ax[1].set(xlabel="days since last contact", title="the real distribution, sentinel removed")
    fig.tight_layout(); return _save(fig, name)


def category_rates(tables, base_rate, name="category_rates.png"):
    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(5.2*n, 4.4), squeeze=False)
    for ax, (col, t) in zip(axes[0], tables.items()):
        t = t.sort_values("rate")
        ax.barh(t[col].astype(str), t["rate"], color=INDIGO,
                xerr=[t["rate"]-t["ci_low"], t["ci_high"]-t["rate"]],
                error_kw={"ecolor": SLATE, "lw": 1})
        ax.axvline(base_rate, ls="--", c=ROSE, lw=1.5)
        ax.set(xlabel="subscription rate", title=col)
        ax.tick_params(labelsize=8)
    fig.suptitle("conversion by controllable lever (bars = 95% Wilson intervals)", fontsize=12)
    fig.tight_layout(); return _save(fig, name)


def fatigue_curve(g, info, name="campaign_fatigue.png"):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(g["campaign"], g["rate"], color=INDIGO, width=.6)
    ax.axhline(info["base_rate"], ls="--", c=ROSE, lw=1.5, label="base rate")
    if info["suggested_stop_after"]:
        ax.axvline(info["suggested_stop_after"] - .5, c=AMBER, lw=2,
                   label=f"stop after {info['suggested_stop_after']}")
    ax.set(xlabel="number of contacts in this campaign", ylabel="subscription rate",
           title="diminishing returns on repeat calls")
    ax.legend(); return _save(fig, name)


def macro_collinearity(df, name="macro_collinearity.png"):
    cols = [c for c in config.MACRO_FEATURES if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="PuOr", center=0, square=True,
                cbar_kws={"shrink": .75}, ax=ax)
    ax.set_title("macro indicators are near-duplicates of each other\n(and a proxy for calendar time)")
    return _save(fig, name)


def roc_pr_comparison(y_true, oof_by_model, name="roc_pr.png"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    base = float(np.mean(y_true))
    for i, (m, p) in enumerate(oof_by_model.items()):
        fpr, tpr, _ = roc_curve(y_true, p)
        ax[0].plot(fpr, tpr, lw=1.8, color=PALETTE[i % len(PALETTE)],
                   label=f"{m} ({auc(fpr,tpr):.4f})")
        pr, rc, _ = precision_recall_curve(y_true, p)
        ax[1].plot(rc, pr, lw=1.8, color=PALETTE[i % len(PALETTE)],
                   label=f"{m} ({average_precision_score(y_true,p):.4f})")
    ax[0].plot([0,1],[0,1],"k--",lw=1,label="chance")
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC, out-of-fold")
    ax[1].axhline(base, ls="--", c="k", lw=1, label=f"chance ({base:.3f})")
    ax[1].set(xlabel="recall", ylabel="precision", title="Precision-Recall, out-of-fold")
    for a in ax: a.legend(fontsize=8)
    fig.tight_layout(); return _save(fig, name)


def profit_plot(curve, best, name="profit_curve.png"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(curve["contacted_frac"]*100, curve["profit"], color=INDIGO, lw=2)
    ax[0].axvline(best["best_contacted_frac"]*100, ls=":", c=AMBER, lw=2,
                  label=f"optimum {best['best_contacted_frac']:.0%}")
    ax[0].axhline(0, c="k", lw=1)
    ax[0].set(xlabel="% of list contacted", ylabel="expected profit",
              title=f"campaign profit  (cost {config.COST_PER_CALL:.0f} / call, "
                    f"value {config.VALUE_PER_CONVERSION:.0f} / conversion)")
    ax[0].legend()
    ax[1].plot(curve["contacted_frac"]*100, curve["precision"], color=INDIGO, label="precision")
    ax[1].plot(curve["contacted_frac"]*100, curve["recall"], color=ROSE, label="recall")
    ax[1].axvline(best["best_contacted_frac"]*100, ls=":", c=AMBER, lw=2)
    ax[1].set(xlabel="% of list contacted", title="precision and recall at each capacity")
    ax[1].legend(); fig.tight_layout(); return _save(fig, name)


def calibration_plot(y_true, oof_by_model, name="calibration.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (m, p) in enumerate(oof_by_model.items()):
        pt, pp = calibration_curve(y_true, p, n_bins=12, strategy="quantile")
        ax.plot(pp, pt, "-o", ms=4, color=PALETTE[i % len(PALETTE)], label=m)
    ax.plot([0,1],[0,1],"k--",lw=1,label="perfect")
    lim = max(.4, max(max(p) for p in oof_by_model.values())*1.05)
    ax.set(xlim=(0,lim), ylim=(0,lim), xlabel="predicted probability",
           ylabel="observed frequency", title="calibration, out-of-fold")
    ax.legend(fontsize=8); return _save(fig, name)


def cv_vs_temporal(tbl, name="cv_vs_temporal.png"):
    d = tbl.sort_values("oof_roc_auc")
    yp = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, max(3.5, .5*len(d))))
    ax.barh(yp-.19, d["oof_roc_auc"], height=.36, color=INDIGO, label="random 5-fold CV")
    ax.barh(yp+.19, d["temporal_roc_auc"], height=.36, color=ROSE, label="temporal holdout")
    ax.set_yticks(yp); ax.set_yticklabels(d["model"])
    ax.set(xlim=(0.5, max(d["oof_roc_auc"].max(), d["temporal_roc_auc"].max())+.03),
           xlabel="ROC-AUC", title="random CV flatters the model; the temporal number is the honest one")
    ax.legend(fontsize=9); return _save(fig, name)


def model_comparison_plot(tbl, name="model_comparison.png"):
    d = tbl.sort_values("oof_roc_auc")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].barh(d["model"], d["oof_roc_auc"], color=INDIGO)
    ax[0].set(title="OOF ROC-AUC", xlim=(max(.5, d["oof_roc_auc"].min()-.03), d["oof_roc_auc"].max()+.01))
    ax[1].barh(d["model"], d["oof_lift_at_10"], color=AMBER)
    ax[1].set(title="lift at top 10% (x base rate)")
    ax[2].barh(d["model"], d["fit_seconds_total"], color=SLATE)
    ax[2].set(title="total fit time (s)")
    fig.tight_layout(); return _save(fig, name)


def confusion_at(y_true, prob, threshold, name="confusion.png"):
    cm = confusion_matrix(y_true, (prob >= threshold).astype(int))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, data, fmt, lab in ((ax[0], cm, "d", "count"),
                              (ax[1], cm/cm.sum(1, keepdims=True)*100, ".1f", "% of actual")):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Purples", cbar=False, ax=a,
                    xticklabels=["pred no", "pred yes"], yticklabels=["actual no", "actual yes"])
        a.set_title(f"threshold {threshold:.3f} ({lab})")
    fig.tight_layout(); return _save(fig, name)


def permutation_importance_plot(result, names, top=20, name="permutation_importance.png"):
    s = pd.Series(result.importances_mean, index=names).sort_values(ascending=False).head(top)
    e = pd.Series(result.importances_std, index=names)[s.index]
    fig, ax = plt.subplots(figsize=(9, max(4, .32*len(s))))
    ax.barh(s.index[::-1], s.values[::-1], xerr=e.values[::-1], color=INDIGO,
            error_kw={"ecolor": SLATE, "lw": 1})
    ax.set(xlabel="drop in ROC-AUC when shuffled", title=f"permutation importance (top {top})")
    ax.tick_params(labelsize=8); return _save(fig, name)
