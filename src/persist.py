"""Saving and loading the production model."""
import json
import joblib
import pandas as pd

from src import config

BUNDLE = config.MODELS_DIR / "model_bundle.joblib"
META = config.MODELS_DIR / "model_meta.json"


def save_bundle(pipeline, input_columns, threshold, metrics=None,
                model_name="model", extra=None):
    """Persist the whole pipeline, the expected input columns, and the threshold.

    The PIPELINE is saved, not the bare classifier. It carries the fitted
    one-hot encoder and scaler, so serving applies exactly the transformations
    training used. Saving only the classifier means rebuilding the encoder at
    serving time from different data, which shifts every column and produces
    confident nonsense with no error.

    The threshold is saved because it came from campaign economics rather than a
    default, and rediscovering it later means guessing.
    """
    joblib.dump({"pipeline": pipeline, "input_columns": list(input_columns),
                 "threshold": float(threshold), "model_name": model_name}, BUNDLE, compress=3)
    META.write_text(json.dumps({
        "model_name": model_name, "n_input_columns": len(input_columns),
        "threshold": float(threshold), "metrics": metrics or {},
        "excluded_features": config.LEAKY_FEATURES, **(extra or {})},
        indent=2), encoding="utf-8")
    return BUNDLE, META


def load_bundle():
    if not BUNDLE.exists():
        raise FileNotFoundError(
            f"{BUNDLE.name} not found. Run the notebook through the "
            f"'save the production model' cell first.")
    b = joblib.load(BUNDLE)
    b["meta"] = json.loads(META.read_text()) if META.exists() else {}
    return b


def score(bundle, frame):
    """Score raw customer rows. Returns (probabilities, decisions)."""
    X = frame.copy()
    for c in config.LEAKY_FEATURES + [config.TARGET]:
        if c in X.columns:
            X = X.drop(columns=[c])
    from src.features import engineer
    X = engineer(X)
    missing = [c for c in bundle["input_columns"] if c not in X.columns]
    if missing:
        raise ValueError(f"missing {len(missing)} required columns: {missing[:6]}")
    prob = bundle["pipeline"].predict_proba(X[bundle["input_columns"]])[:, 1]
    return prob, (prob >= bundle["threshold"]).astype(int)
