"""Optional extra: a call-list scoring service.

The brief does not require deployment. This exists because the deliverable the
marketing team actually wants is a ranked call list, not a notebook.
"""
import io
import logging

import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash

from src import config
from src.persist import load_bundle, score

app = Flask(__name__)
app.config.update(SECRET_KEY="change-me-in-production",
                  MAX_CONTENT_LENGTH=32 * 1024 * 1024)
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(config.LOGS_DIR / "app.log"),
                              logging.StreamHandler()])
log = logging.getLogger("pbank")

_BUNDLE = None


def bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_bundle()
        log.info("loaded %s, threshold %.4f", _BUNDLE["model_name"], _BUNDLE["threshold"])
    return _BUNDLE


@app.route("/")
def index():
    try:
        return render_template("index.html", meta=bundle()["meta"], error=None)
    except FileNotFoundError as exc:
        return render_template("index.html", meta={}, error=str(exc))


@app.route("/score", methods=["POST"])
def score_csv():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Choose a customer CSV to score."); return redirect(url_for("index"))
    if not f.filename.lower().endswith(".csv"):
        flash("Upload a .csv file."); return redirect(url_for("index"))
    try:
        raw = f.read()
        frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        prob, decision = score(bundle(), frame)
    except ValueError as exc:
        flash(str(exc)); return redirect(url_for("index"))
    except Exception as exc:
        log.exception("scoring failed")
        flash(f"Could not score that file: {exc}"); return redirect(url_for("index"))

    show = [c for c in ("age", "job", "contact", "month", "campaign", "poutcome")
            if c in frame.columns]
    out = frame[show].copy()
    out["probability"] = prob.round(4)
    out["call"] = decision
    out = out.sort_values("probability", ascending=False)
    log.info("scored %d rows, %d on the call list", len(out), int(decision.sum()))
    return render_template("results.html", rows=out.head(200).to_dict("records"),
                           columns=show, total=len(out), flagged=int(decision.sum()),
                           threshold=bundle()["threshold"])


@app.route("/api/score", methods=["POST"])
def api_score():
    payload = request.get_json(silent=True)
    if not payload or "rows" not in payload:
        return jsonify(error="POST {'rows': [{column: value, ...}]}"), 400
    try:
        prob, decision = score(bundle(), pd.DataFrame(payload["rows"]))
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(threshold=bundle()["threshold"],
                   results=[{"probability": round(float(p), 4), "call": int(d)}
                            for p, d in zip(prob, decision)])


@app.route("/health")
def health():
    ok = (config.MODELS_DIR / "model_bundle.joblib").exists()
    return jsonify(status="ok" if ok else "model_missing",
                   loaded=_BUNDLE is not None), (200 if ok else 503)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
