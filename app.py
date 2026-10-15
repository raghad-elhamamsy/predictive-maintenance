"""Signal Monitor — Flask backend for the predictive-maintenance dashboard.

Signal Monitor is an experimental research prototype for early warning on a
signal asset (display label ``asset monitored asset asset``).

Design rules enforced here
-------------------------
* The dashboard uses **real historical model inference** by default: it scores
  the local cleaned history with the local model bundles via ``src/historical``.
  No environment variable is required to start the real app.
* It is local-only (``127.0.0.1``) and reads only local artifacts. If the
  required artifacts are missing or invalid, endpoints return a clear, safe JSON
  error; the app never substitutes fabricated or demo data.
* **Model Insights** loads the supervised bundles read-only for identity and
  limitations, plus safe aggregate report metrics. Nothing is retrained.
* No raw readings, private channel identifier names, filesystem paths, stack traces, or model
  files are ever returned by an endpoint. No directory listing or debug mode.

Run::

    python app.py            # http://127.0.0.1:5000
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 5000

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Local artifact discovery (read-only; never surfaced as paths)
# ---------------------------------------------------------------------------
def _bundle_files() -> list[Path]:
    out = []
    for name in ("05_logreg_early_warning.joblib", "05_tree_early_warning.joblib"):
        p = ROOT / "models" / name
        if p.is_file():
            out.append(p)
    return out


def _reports_dir() -> Path:
    return ROOT / "reports" / "tables"


def _historical_artifacts() -> dict:
    d = _reports_dir()
    return {
        "availability": (d / "08_data_availability.csv") if (d / "08_data_availability.csv").is_file() else None,
        "episodes": (d / "08_alarm_episodes_union.csv") if (d / "08_alarm_episodes_union.csv").is_file() else None,
        "supervised": (d / "05_supervised_summary.csv") if (d / "05_supervised_summary.csv").is_file() else None,
        "leaderboard": (d / "07_comparison_leaderboard.csv") if (d / "07_comparison_leaderboard.csv").is_file() else None,
    }


# ---------------------------------------------------------------------------
# Safe dataframe readers (aggregate reports only; never raw system numbers)
# ---------------------------------------------------------------------------
def _read_csv_columns(path: Path, columns: list[str]) -> list[dict]:
    """Read only the named columns from an aggregate report; drop unknown ones."""
    import pandas as pd

    df = pd.read_csv(path)
    keep = [c for c in columns if c in df.columns]
    if not keep:
        return []
    df = df[keep]
    df = df.where(df.notna(), None)
    return json.loads(df.to_json(orient="records"))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the Signal Monitor dashboard shell."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Liveness probe."""
    return jsonify(status="ok", service="signal_monitor", mode="historical")


@app.route("/api/modes")
def api_modes():
    """Startup status: whether real historical scoring is ready, and the models."""
    from src import historical as hsvc

    models = hsvc.available_models()
    scoring_ready = bool(hsvc.CLEAN_PATH.is_file() and hsvc.LABELS_PATH.is_file() and models)
    default_model = hsvc.DEFAULT_MODEL if hsvc.DEFAULT_MODEL in models else (models[0] if models else None)

    return jsonify(
        mode="historical",
        ready=scoring_ready,
        models=[{"id": mid, "label": hsvc.MODEL_LABELS.get(mid, mid)} for mid in models],
        default_model=default_model,
        detail=_historical_detail(scoring_ready),
    )


def _historical_detail(scoring_ready: bool) -> str:
    if not scoring_ready:
        return ("The required local scoring artifacts (a model bundle and the cleaned "
                "history) are not present on this machine.")
    return "Scoring the local historical data with the selected model."


_HIST_ERR_STATUS = {
    "bad_model": 400,
    "model_missing": 503,
    "data_missing": 503,
    "model_invalid": 422,
    "data_invalid": 422,
    "align_mismatch": 422,
    "insufficient_history": 422,
    "scoring_failed": 500,
}


def _parse_days():
    """Validate the optional ``days`` query parameter (1..3650)."""
    raw = request.args.get("days")
    if raw is None or raw == "":
        return None, None
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return None, (jsonify(error="bad_range", message="The 'days' parameter must be an integer."), 400)
    if days < 1 or days > 3650:
        return None, (
            jsonify(error="bad_range", message="The 'days' parameter must be between 1 and 3650."),
            400,
        )
    return days, None


@app.route("/api/model/insights")
def api_model_insights():
    """Model identity, aggregate metrics and limitations (read-only)."""
    bundles = _bundle_files()
    art = _historical_artifacts()

    models = []
    for path in bundles:
        info = _bundle_identity(path)
        if info is not None:
            models.append(info)

    metrics = _aggregate_metrics(art)

    return jsonify(
        source={
            "id": "model",
            "label": "Model Insights",
            "detail": "Local model bundles and aggregate reports; no inference is run.",
            "live": False,
        },
        bundles_present=bool(models),
        models=models,
        metrics=metrics,
        limitations=_LIMITATIONS,
    )


_LIMITATIONS = [
    "Supervised performance is not validated: the dataset contains a single provisional failure event.",
    "The provisional failure date is a placeholder, not confirmed ground truth.",
    "There is no live system context; historical scoring is offline data.",
    "An alarm is a statistical indication only and does not confirm a failure.",
    "Historical sensor traces are shown as already-scaled values under generic labels.",
]


def _bundle_identity(path: Path) -> dict | None:
    """Load a bundle read-only and report identity; never expose internals."""
    try:
        import joblib

        bundle = joblib.load(path)
        from src.scoring import assert_bundle_complete, bundle_core

        assert_bundle_complete(bundle)
        core = bundle_core(bundle)
        scaler = bundle["sensor_scaler"]
        return {
            "id": _bundle_model_id(path.name),
            "name": _pretty_bundle_name(path.name),
            "horizon_days": int(bundle["horizon"]),
            "threshold": float(bundle["threshold"]),
            "sustain_days": int(bundle["sustain_k"]),
            "n_features": len(bundle["features"]),
            "n_core_sensors": len(core),
            "baseline_year": int(scaler["baseline_year"]),
            "baseline_rows": int(scaler["n_baseline_rows"]),
            "note": str(bundle.get("note", "")),
        }
    except Exception:  # noqa: BLE001 - never leak internals to the client
        return None


def _pretty_bundle_name(filename: str) -> str:
    if "logreg" in filename:
        return "Logistic regression (supervised)"
    if "tree" in filename:
        return "Decision tree (supervised)"
    return "Supervised model"


def _bundle_model_id(filename: str) -> str:
    if "logreg" in filename:
        return "logreg"
    if "tree" in filename:
        return "tree"
    return filename


def _aggregate_metrics(art: dict) -> list[dict]:
    """Safe aggregate metrics from reports/tables, when present."""
    out: list[dict] = []
    if art["supervised"]:
        rows = _read_csv_columns(
            art["supervised"],
            ["model", "scored_rows", "scored_pos", "total_pos", "prevalence", "PR_AUC", "ROC_AUC"],
        )
        for r in rows:
            out.append({"group": "supervised summary", **r})
    if art["leaderboard"]:
        rows = _read_csv_columns(
            art["leaderboard"],
            ["approach", "model", "alarm_setting", "lead_days",
             "false_alarm_days_per_month", "sustained_alarm_share_2033"],
        )
        for r in rows:
            out.append({"group": "comparison leaderboard", **r})
    return out


@app.route("/api/historical/summary")
def api_historical_summary():
    """Approved aggregate historical view (read-only, local artifacts)."""
    art = _historical_artifacts()
    if not (art["availability"] and art["episodes"]):
        return jsonify(
            error="artifacts_missing",
            message=("Historical mode is enabled but the required aggregate report artifacts "
                     "are not present on this machine. Generate them with the project notebooks, "
                     "then restart Signal Monitor."),
        ), 503

    periods = _read_csv_columns(art["availability"], ["period", "start", "end", "days", "role"])
    episodes_raw = _read_csv_columns(
        art["episodes"], ["start", "end", "days", "resolved"]
    )
    episodes = []
    for i, e in enumerate(episodes_raw, start=1):
        episodes.append({
            "id": f"HIST-{i:02d}",
            "start": e.get("start"),
            "end": e.get("end"),
            "days": e.get("days"),
            "resolved": bool(e.get("resolved")) if e.get("resolved") is not None else None,
        })

    return jsonify(
        source={
            "id": "historical",
            "label": "Historical system Data",
            "detail": "Offline historical records. Not a live context.",
            "live": False,
        },
        disclaimer=("Historical offline records. Provisional failure labels are not confirmed "
                    "ground truth. Sensor traces are offered only as scaled values under generic "
                    "labels via /api/historical/series."),
        periods=periods,
        episodes=episodes,
        signal_trends_available=True,
    )


@app.route("/api/historical/scores")
def api_historical_scores():
    """Real model scores over the local historical history."""
    from src import historical

    days, err = _parse_days()
    if err is not None:
        return err
    model_id = request.args.get("model", historical.DEFAULT_MODEL)
    if model_id not in historical.MODEL_PATHS:
        return jsonify(error="bad_model", message="Unknown model identifier."), 400

    try:
        payload = historical.score_history(model_id, days=days)
    except historical.HistoricalError as exc:
        return jsonify(error=exc.code, message=exc.message), _HIST_ERR_STATUS.get(exc.code, 500)
    return jsonify(payload)


@app.route("/api/historical/series")
def api_historical_series():
    """Scaled historical sensor traces under generic labels."""
    from src import historical

    days, err = _parse_days()
    if err:
        return err
    model_id = request.args.get("model", historical.DEFAULT_MODEL)
    if model_id not in historical.MODEL_PATHS:
        return jsonify(error="bad_model", message="Unknown model identifier."), 400
    try:
        return jsonify(historical.series_history(days=days, model_id=model_id))
    except historical.HistoricalError as exc:
        status = _HIST_ERR_STATUS.get(exc.code, 500)
        return jsonify(error=exc.code, message=exc.message), status


# ---------------------------------------------------------------------------
# Safe error handling (no stack traces, no paths)
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def _not_found(_err):
    return jsonify(error="not_found", message="The requested resource does not exist."), 404


@app.errorhandler(400)
def _bad_request(_err):
    return jsonify(error="bad_request", message="The request was not valid."), 400


@app.errorhandler(500)
def _server_error(_err):
    return jsonify(error="server_error", message="An internal error occurred."), 500


@app.errorhandler(Exception)
def _unhandled(_err):
    return jsonify(error="server_error", message="An internal error occurred."), 500


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
