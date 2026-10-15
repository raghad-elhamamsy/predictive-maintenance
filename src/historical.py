"""Read-only historical scoring support for the Signal Monitor dashboard.

This module is the single place that turns the project's existing local
artifacts into dashboard-ready results. It does **not** re-implement any
scoring or feature engineering: it loads the cleaned sensor history and the
aligned segment series and delegates to the canonical ``src.scoring.score_new``.

Design rules
------------
* Read only: it never writes to the repository.
* Privacy: it never returns raw readings, real sensor/channel identifier names, filesystem
  paths or model-bundle contents. Sensor series are exposed only under generic
  display labels (``Signal 01`` ...) using already-scaled values.
* Honest: missing scores are returned as ``None`` (JSON ``null``), never NaN or
  infinity; failures raise :class:`HistoricalError` with a safe message.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring import assert_bundle_complete, bundle_core, score_new

ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = ROOT / "data" / "interim" / "channel_clean.csv"
LABELS_PATH = ROOT / "data" / "processed" / "features_labels.csv"
SCALED_PATH = ROOT / "data" / "interim" / "channel_scaled.csv"

MODEL_PATHS = {
    "logreg": ROOT / "models" / "05_logreg_early_warning.joblib",
    "tree": ROOT / "models" / "05_tree_early_warning.joblib",
}
MODEL_LABELS = {
    "logreg": "Logistic regression (supervised)",
    "tree": "Decision tree (supervised)",
}
DEFAULT_MODEL = "logreg"

MIN_ROWS = 30
MAX_DAYS = 3650

_COLORS = ["#2dd4bf", "#5aa9ff", "#f0b429", "#a78bfa", "#f0616d",
           "#22d3ee", "#facc15", "#34d399", "#c084fc", "#fb923c",
           "#38bdf8", "#4ade80", "#e879f9", "#f87171"]


class HistoricalError(Exception):
    """A safe, user-facing historical-mode failure.

    ``code`` maps to an HTTP status by the Flask layer; ``message`` is safe to
    show and contains no paths or system values.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def available_models(model_paths: dict[str, Path] | None = None) -> list[str]:
    """Model ids whose bundle file exists locally (order preserved)."""
    paths = model_paths or MODEL_PATHS
    return [mid for mid, p in paths.items() if Path(p).is_file()]


def load_bundle(model_id: str = DEFAULT_MODEL, model_paths: dict[str, Path] | None = None):
    """Load and validate a bundle read-only, or raise :class:`HistoricalError`."""
    paths = model_paths or MODEL_PATHS
    if model_id not in paths:
        raise HistoricalError("bad_model", "Unknown model identifier.")
    path = Path(paths[model_id])
    if not path.is_file():
        raise HistoricalError("model_missing", "The selected model bundle is not available on this machine.")
    try:
        import joblib

        bundle = joblib.load(path)
        assert_bundle_complete(bundle)
    except HistoricalError:
        raise
    except Exception as exc:  # noqa: BLE001 - never leak internals
        raise HistoricalError("model_invalid", "The selected model bundle could not be validated.") from exc
    return bundle


def _safe_float(value) -> float | None:
    """A JSON-safe float: None for NaN/inf, otherwise a plain float."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _sorted_aligned(clean_path: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load and validate the cleaned history and its aligned segment ids."""
    if not Path(clean_path).is_file():
        raise HistoricalError("data_missing", "The cleaned historical data file is not available on this machine.")
    if not Path(labels_path).is_file():
        raise HistoricalError("data_missing", "The feature/label artifact with segment ids is not available on this machine.")

    try:
        clean = pd.read_csv(clean_path)
        labels = pd.read_csv(labels_path)
    except Exception as exc:  # noqa: BLE001
        raise HistoricalError("data_invalid", "The historical artifacts could not be read.") from exc

    for name, frame in (("clean history", clean), ("segment artifact", labels)):
        if "timestamp" not in frame.columns:
            raise HistoricalError("data_invalid", f"The {name} is missing a timestamp column.")

    stamps = pd.to_datetime(clean["timestamp"], errors="coerce")
    if stamps.isna().any():
        raise HistoricalError("data_invalid", "The historical data contains unparseable timestamps.")

    if len(clean) != len(labels):
        raise HistoricalError("align_mismatch", "The historical data and segment artifact have different row counts.")

    lstamps = pd.to_datetime(labels["timestamp"], errors="coerce")
    if lstamps.isna().any():
        raise HistoricalError("data_invalid", "The segment artifact contains unparseable timestamps.")
    if not stamps.reset_index(drop=True).equals(lstamps.reset_index(drop=True)):
        raise HistoricalError("align_mismatch", "The historical data and segment artifact are not aligned by timestamp.")

    if "segment_id" not in labels.columns:
        raise HistoricalError("data_invalid", "The segment artifact is missing the segment id column.")

    order = np.argsort(stamps.to_numpy())
    clean = clean.iloc[order].reset_index(drop=True)
    seg = labels["segment_id"].iloc[order].reset_index(drop=True)
    stamps = stamps.iloc[order].reset_index(drop=True)

    if len(clean) < MIN_ROWS:
        raise HistoricalError("insufficient_history",
                              f"At least {MIN_ROWS} historical rows are required to build the rolling features.")

    clean.index = pd.DatetimeIndex(stamps)
    seg.index = pd.DatetimeIndex(stamps)
    coverage = {
        "start": stamps.iloc[0].strftime("%Y-%m-%d"),
        "end": stamps.iloc[-1].strftime("%Y-%m-%d"),
        "days": int(len(clean)),
    }
    return clean, seg, coverage


def _alarm_episodes(points: list[dict], model_label: str) -> list[dict]:
    """Contiguous runs of alarmed days, as safe model-generated episodes."""
    episodes: list[dict] = []
    run: dict | None = None
    for pt in points:
        if pt["alarm"]:
            if run is None:
                run = {"start": pt["timestamp"], "end": pt["timestamp"], "days": 1,
                       "peak": pt["probability"] or 0.0}
            else:
                run["end"] = pt["timestamp"]
                run["days"] += 1
                run["peak"] = max(run["peak"], pt["probability"] or 0.0)
        elif run is not None:
            episodes.append(run)
            run = None
    if run is not None:
        episodes.append(run)
    for i, ep in enumerate(episodes, start=1):
        ep["id"] = f"MA-{i:02d}"
        ep["model"] = model_label
        ep["kind"] = "model_alarm"
    return episodes


def score_history(model_id: str = DEFAULT_MODEL, days: int | None = None, *,
                  clean_path: Path = CLEAN_PATH, labels_path: Path = LABELS_PATH,
                  model_paths: dict[str, Path] | None = None) -> dict:
    """Score the local historical history and return a safe dashboard payload."""
    clean_path = Path(clean_path)
    labels_path = Path(labels_path)
    if not clean_path.is_file():
        raise HistoricalError("data_missing", "The cleaned historical data file is not available on this machine.")
    if not labels_path.is_file():
        raise HistoricalError("data_missing", "The feature/label artifact with segment ids is not available on this machine.")

    bundle = load_bundle(model_id, model_paths)
    clean, seg, coverage = _sorted_aligned(clean_path, labels_path)

    try:
        result = score_new(clean, seg, bundle)
    except HistoricalError:
        raise
    except Exception as exc:  # noqa: BLE001 - never leak internals
        raise HistoricalError("scoring_failed", "Historical scoring failed for the local artifacts.") from exc

    full_points = []
    for ts, row in result.iterrows():
        full_points.append({
            "timestamp": pd.Timestamp(ts).strftime("%Y-%m-%d"),
            "probability": _safe_float(row["probability"]),
            "alarm": bool(row["alarm"]),
        })

    points = full_points
    if days is not None:
        points = full_points[-int(days):]

    scored = sum(1 for p in points if p["probability"] is not None)
    alarms = sum(1 for p in points if p["alarm"])
    last_scored = next((p["timestamp"] for p in reversed(points) if p["probability"] is not None), None)
    model_label = MODEL_LABELS.get(model_id, "Supervised model")

    return {
        "source": {
            "id": "historical",
            "label": "Historical system Data",
            "detail": "Offline records scored by the local model. Not a live context.",
            "live": False,
        },
        "model": {
            "id": model_id,
            "label": model_label,
            "horizon_days": int(bundle["horizon"]),
            "threshold": _safe_float(bundle["threshold"]),
            "sustain_days": int(bundle["sustain_k"]),
        },
        "coverage": coverage,
        "rows": len(points),
        "scored": scored,
        "unscored": len(points) - scored,
        "alarms": alarms,
        "last_processed": full_points[-1]["timestamp"] if full_points else None,
        "last_scored": next((p["timestamp"] for p in reversed(points) if p["probability"] is not None), None),
        "points": points,
        "episodes": _alarm_episodes(points, MODEL_LABELS.get(model_id, "Supervised model")),
    }


def series_history(days: int | None = None, *, model_id: str = DEFAULT_MODEL,
                   scaled_path: Path = SCALED_PATH,
                   model_paths: dict[str, Path] | None = None) -> dict:
    """Scaled historical sensor traces under generic display labels."""
    scaled_path = Path(scaled_path)
    if not scaled_path.is_file():
        raise HistoricalError("data_missing", "The scaled historical data file is not available on this machine.")

    bundle = load_bundle(model_id, model_paths)
    try:
        scaled = pd.read_csv(scaled_path)
    except Exception as exc:  # noqa: BLE001
        raise HistoricalError("data_invalid", "The scaled historical data could not be read.") from exc
    if "timestamp" not in scaled.columns:
        raise HistoricalError("data_invalid", "The scaled historical data is missing a timestamp column.")

    stamps = pd.to_datetime(scaled["timestamp"], errors="coerce")
    if stamps.isna().any():
        raise HistoricalError("data_invalid", "The scaled historical data contains unparseable timestamps.")

    core = bundle_core(bundle)
    missing = [c for c in core if c not in scaled.columns]
    if missing:
        raise HistoricalError("data_invalid", "The scaled historical data is missing required sensor columns.")

    order = np.argsort(stamps.to_numpy())
    scaled = scaled.iloc[order].reset_index(drop=True)
    stamps = stamps.iloc[order].reset_index(drop=True)
    if days is not None:
        scaled = scaled.tail(int(days)).reset_index(drop=True)
        stamps = stamps.tail(int(days)).reset_index(drop=True)

    sensors = []
    values: dict[str, list] = {}
    for i, col in enumerate(core):
        key = f"signal_{i + 1:02d}"
        sensors.append({
            "key": key,
            "label": f"Signal {i + 1:02d}",
            "unit": "scaled",
            "color": _COLORS[i % len(_COLORS)],
        })
        values[key] = [_safe_float(v) for v in scaled[col].tolist()]

    dates = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in stamps]
    coverage = {"start": dates[0] if dates else None, "end": dates[-1] if dates else None, "days": len(dates)}
    return {
        "source": {
            "id": "historical",
            "label": "Historical system Data",
            "detail": "Scaled sensor history under generic labels. Not a live context.",
            "live": False,
        },
        "labeling": "generic",
        "note": ("Series are already-scaled core sensors shown under generic labels. "
                 "Real system channel identifier names and raw values are never exposed."),
        "coverage": coverage,
        "dates": dates,
        "sensors": sensors,
        "values": values,
    }
