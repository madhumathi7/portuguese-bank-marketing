"""Task 3: turn the model into advice the marketing team can act on.

THE RULE THAT MAKES THIS USEFUL. Split features into what the bank controls and
what it does not. Age, job and the euribor rate are strong predictors and utterly
useless as advice -- "target younger customers when interest rates are low" is an
observation, not an action the campaign team can take next Monday. Contact
channel, month, day of week and how many times to call are all decisions someone
actually makes, so that is where recommendations must come from.
"""
import numpy as np
import pandas as pd

from src import config


def rate_by_category(df, column, target=config.TARGET, min_count=50):
    """Subscription rate per level, with counts and a Wilson interval.

    The interval is not decoration. A 40% success rate on 12 customers is noise,
    and without an interval it will end up in someone's slide deck as a strategy.
    """
    y = df[target] if pd.api.types.is_numeric_dtype(df[target]) else \
        (df[target].astype("string").str.lower() == "yes").astype(int)
    g = (pd.DataFrame({column: df[column], "y": y}).groupby(column)["y"]
         .agg(n="count", conversions="sum", rate="mean").reset_index())
    g = g[g["n"] >= min_count].copy()

    z, p, n = 1.96, g["rate"], g["n"]
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    g["ci_low"], g["ci_high"] = centre - margin, centre + margin
    g["lift_vs_base"] = g["rate"] / y.mean()
    return g.sort_values("rate", ascending=False).reset_index(drop=True)


def campaign_fatigue(df, target=config.TARGET, max_contacts=12):
    """Success rate against number of contacts. The clearest actionable finding.

    Expect sharp diminishing returns. Every call after the point where the rate
    collapses costs money and annoys a customer who has already said no, so this
    curve translates directly into a stopping rule.
    """
    y = df[target] if pd.api.types.is_numeric_dtype(df[target]) else \
        (df[target].astype("string").str.lower() == "yes").astype(int)
    d = pd.DataFrame({"campaign": df["campaign"].clip(upper=max_contacts), "y": y})
    g = (d.groupby("campaign")["y"].agg(n="count", conversions="sum", rate="mean")
         .reset_index())
    g["cumulative_share_of_calls"] = g["n"].cumsum() / g["n"].sum()
    base = float(y.mean())
    below = g[(g["rate"] < base * 0.5) & (g["n"] >= 50)]
    stop_at = int(below["campaign"].iloc[0]) if len(below) else None
    return g, {"base_rate": round(base, 4), "suggested_stop_after": stop_at,
               "share_of_calls_beyond_stop": (
                   round(float(g.loc[g["campaign"] >= stop_at, "n"].sum() / g["n"].sum()), 4)
                   if stop_at else None)}


def controllable_summary(df, target=config.TARGET):
    """Rate tables for every lever the bank actually pulls."""
    return {c: rate_by_category(df, c, target)
            for c in config.CONTROLLABLE_FEATURES
            if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])}


def build_recommendations(df, target=config.TARGET, importance=None):
    """Assemble ranked, quantified suggestions. Each carries its own evidence."""
    y = df[target] if pd.api.types.is_numeric_dtype(df[target]) else \
        (df[target].astype("string").str.lower() == "yes").astype(int)
    base = float(y.mean())
    recs = []

    if "contact" in df:
        t = rate_by_category(df, "contact", target)
        if len(t) > 1:
            best, worst = t.iloc[0], t.iloc[-1]
            recs.append({
                "lever": "Contact channel",
                "action": f"Route the campaign through {best['contact']} rather than {worst['contact']}",
                "evidence": (f"{best['rate']:.1%} vs {worst['rate']:.1%} conversion "
                             f"({best['lift_vs_base']:.2f}x vs base)"),
                "confidence": ("high" if best["ci_low"] > worst["ci_high"]
                               else "low - intervals overlap")})

    if "month" in df:
        t = rate_by_category(df, "month", target)
        top = t.head(3)
        recs.append({
            "lever": "Timing",
            "action": f"Concentrate volume in {', '.join(top['month'])}",
            "evidence": (f"{top['rate'].mean():.1%} average conversion vs "
                         f"{base:.1%} overall"),
            "confidence": "high" if (top["ci_low"] > base).all() else "moderate"})

    if "campaign" in df:
        _, fat = campaign_fatigue(df, target)
        if fat["suggested_stop_after"]:
            recs.append({
                "lever": "Contact frequency",
                "action": f"Stop after {fat['suggested_stop_after']} attempts on the same customer",
                "evidence": (f"conversion falls below half the base rate beyond that "
                             f"point; {fat['share_of_calls_beyond_stop']:.1%} of all "
                             f"calls are currently spent there"),
                "confidence": "high"})

    if "poutcome" in df:
        t = rate_by_category(df, "poutcome", target)
        succ = t[t["poutcome"] == "success"]
        if len(succ):
            s = succ.iloc[0]
            recs.append({
                "lever": "Targeting",
                "action": "Prioritise customers who accepted a previous campaign",
                "evidence": f"{s['rate']:.1%} conversion, {s['lift_vs_base']:.1f}x base rate",
                "confidence": "high"})

    recs.append({
        "lever": "Measurement",
        "action": "Hold out a randomised control group from the next campaign",
        "evidence": ("every figure here is observational, so it shows association "
                     "and cannot prove that changing the lever causes the lift"),
        "confidence": "n/a - this is how you get causal evidence"})
    return pd.DataFrame(recs)
