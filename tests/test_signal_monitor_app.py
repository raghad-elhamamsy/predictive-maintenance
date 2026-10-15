"""
Focused tests for the Signal Monitor Flask backend (``app.py``).

Plain assert-based script -- no pytest required. Run from the repository root:

    python tests/test_signal_monitor_app.py

It exercises the public HTTP surface with Flask's test client and checks that:

    * the app starts in local historical scoring mode (no environment variable)
        and reports its readiness via ``/api/modes``;
    * missing local artifacts produce a clear, safe
    JSON error instead of substituting any synthetic/demo data;
    * warning dates are derived exclusively from local dataset timestamps
    (no hardcoded or fabricated years);
    * no endpoint leaks raw readings, source identifiers, filesystem paths or stack
    traces; unknown routes and errors come back as safe JSON, never a debug page.

No private artifact is required: historical checks degrade gracefully when the
gitignored bundles or data files are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as app_module  # noqa: E402
from src import historical as hist  # noqa: E402

FORBIDDEN = (
    "leading_sensors",
    "interpretation",
    "operating_state",
    "Traceback",
    "C:\\Users",
    ".xlsx",
    "joblib",
)


def _no_leaks(label: str, text: str) -> None:
    hits = [tok for tok in FORBIDDEN if tok in text]
    assert not hits, f"{label}: forbidden token(s) present: {hits}"


def check_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert body["status"] == "ok", body
    assert body["service"] == "signal_monitor", body
    assert body["mode"] == "historical", body
    _no_leaks("/health", r.get_data(as_text=True))


def check_index(client) -> None:
    r = client.get("/")
    assert r.status_code == 200, r.status_code
    html = r.get_data(as_text=True)
    for anchor in ("rangeSelector", "rangeSelector2", "summaryCards",
                   "modelSelector", "warningTable", "app.js"):
        assert anchor in html, f"shell missing {anchor}"
    assert "SYNTHETIC" not in html, "page must not advertise a synthetic mode"
    assert "2030" not in html, "page must not contain fabricated dates"
    _no_leaks("/", html)


def check_demo_removed(client) -> None:
    r = client.get("/api/demo")
    assert r.status_code == 404, r.status_code
    body = r.get_json()
    assert body is not None and body.get("error") == "not_found", body


def check_modes_auto(client) -> None:
    """Historical mode is the automatic default; no env flag is required."""
    r = client.get("/api/modes")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert body["mode"] == "historical", body
    assert isinstance(body["ready"], bool), body
    assert isinstance(body["models"], list), body["models"]
    assert "default_model" in body, body
    assert isinstance(body["detail"], str) and body["detail"], body
    if body["ready"]:
        assert body["models"], "ready implies at least one model"
        ids = [m["id"] for m in body["models"]]
        assert body["default_model"] in ids, body["default_model"]
    _no_leaks("/api/modes", r.get_data(as_text=True))


def check_insights(client) -> None:
    r = client.get("/api/model/insights")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert isinstance(body["bundles_present"], bool), body["bundles_present"]
    assert isinstance(body["models"], list), body["models"]
    assert isinstance(body["metrics"], list), body["metrics"]
    assert body["limitations"], "limitations must not be empty"
    assert body["source"]["id"] == "model", body["source"]
    assert "illustrative" not in body, "synthetic illustrative models must be gone"
    _no_leaks("/api/model/insights", r.get_data(as_text=True))


def check_historical_always_on(client) -> None:
    """Without any env flag the endpoints answer (never 403 'disabled')."""
    for path in ("/api/historical/summary", "/api/historical/scores",
                 "/api/historical/series"):
        r = client.get(path)
        assert r.status_code != 403, (path, r.status_code)
        assert r.status_code in (200, 422, 503), (path, r.status_code)
        _no_leaks(path, r.get_data(as_text=True))

    # Bad query parameters are rejected safely, regardless of artifacts.
    assert client.get("/api/historical/scores?model=nope").status_code == 400
    assert client.get("/api/historical/scores?days=abc").status_code == 400
    assert client.get("/api/historical/series?days=abc").status_code == 400


def check_scores_shape_and_dates(client) -> None:
    """Real scores, and every warning date comes from the dataset itself."""
    sc = client.get("/api/historical/scores?model=logreg")
    assert sc.status_code in (200, 422, 503), sc.status_code
    if sc.status_code != 200:
        return
    body = sc.get_json()
    assert body["source"]["id"] == "historical"
    assert body["model"]["id"] == "logreg"
    assert isinstance(body["points"], list) and body["points"]
    assert body["scored"] <= body["rows"]
    assert body["alarms"] <= body["scored"]
    assert set(body["coverage"]) == {"start", "end", "days"}

    for pt in body["points"]:
        assert set(pt) == {"timestamp", "probability", "alarm"}, pt
        assert pt["probability"] is None or isinstance(pt["probability"], (int, float))
        assert not (isinstance(pt["probability"], float) and pt["probability"] != pt["probability"])
        assert isinstance(pt["alarm"], bool)

    stamps = [p["timestamp"] for p in body["points"]]
    stamp_set = set(stamps)
    assert stamps == sorted(stamps), "points must be time-ordered"
    assert body["coverage"]["start"] == stamps[0], body["coverage"]
    assert body["coverage"]["end"] == stamps[-1], body["coverage"]

    for ep in body["episodes"]:
        assert set(ep).issubset({"id", "start", "end", "days", "peak", "model", "kind"}), ep
        assert ep["kind"] == "model_alarm", ep
        assert ep["start"] in stamp_set, f"{ep['id']} start is not a dataset timestamp"
        assert ep["end"] in stamp_set, f"{ep['id']} end is not a dataset timestamp"
        assert ep["start"] <= ep["end"], ep
    _no_leaks("/api/historical/scores", sc.get_data(as_text=True))

    se = client.get("/api/historical/series")
    assert se.status_code in (200, 422, 503), se.status_code
    if se.status_code == 200:
        sbody = se.get_json()
        assert sbody["labeling"] == "generic"
        assert len(sbody["sensors"]) == len(sbody["values"])
        assert len(sbody["dates"]) == sbody["coverage"]["days"]
        for i, s in enumerate(sbody["sensors"], start=1):
            assert s["label"] == f"Signal {i:02d}", s
            assert len(sbody["values"][s["key"]]) == len(sbody["dates"])
        _no_leaks("/api/historical/series", se.get_data(as_text=True))


def check_missing_artifacts(client) -> None:
    """Missing local artifacts produce a clear, safe error (no demo fallback)."""
    missing = ROOT / "data" / "does_not_exist_artifact.csv"
    saved = {
        "CLEAN_PATH": hist.CLEAN_PATH,
        "LABELS_PATH": hist.LABELS_PATH,
        "SCALED_PATH": hist.SCALED_PATH,
        "score_history": hist.score_history,
        "series_history": hist.series_history,
    }
    orig_score = saved["score_history"]
    orig_series = saved["series_history"]
    try:
        hist.CLEAN_PATH = missing
        hist.LABELS_PATH = missing
        hist.SCALED_PATH = missing
        hist.score_history = lambda model_id=hist.DEFAULT_MODEL, days=None: orig_score(
            model_id, days=days, clean_path=missing, labels_path=missing)
        hist.series_history = lambda days=None, model_id=hist.DEFAULT_MODEL: orig_series(
            days=days, model_id=model_id, scaled_path=missing)

        modes = client.get("/api/modes").get_json()
        assert modes["ready"] is False, modes
        assert modes["detail"], modes

        sc = client.get("/api/historical/scores?model=logreg")
        assert sc.status_code == 503, sc.status_code
        assert sc.get_json()["error"] == "data_missing", sc.get_json()
        _no_leaks("/api/historical/scores (missing)", sc.get_data(as_text=True))

        se = client.get("/api/historical/series?model=logreg")
        assert se.status_code == 503, se.status_code
        assert se.get_json()["error"] == "data_missing", se.get_json()
    finally:
        hist.CLEAN_PATH = saved["CLEAN_PATH"]
        hist.LABELS_PATH = saved["LABELS_PATH"]
        hist.SCALED_PATH = saved["SCALED_PATH"]
        hist.score_history = orig_score
        hist.series_history = orig_series


def check_no_fabricated_dates_in_source() -> None:
    """The code itself must not hardcode a fake year or synthetic fallback."""
    for rel in ("app.py", "static/js/app.js"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "2030" not in text, f"{rel} contains a fabricated year"
        assert "SYNTHETIC" not in text and "synthetic" not in text, f"{rel} still references synthetic data"
        assert "mulberry32" not in text, f"{rel} still contains the fallback generator"


def check_safe_errors(client) -> None:
    r = client.get("/this-route-does-not-exist")
    assert r.status_code == 404, r.status_code
    body = r.get_json()
    assert body is not None and body.get("error") == "not_found", body
    _no_leaks("404", r.get_data(as_text=True))


def main() -> int:
    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()

    checks = [
        ("health", check_health),
        ("index shell", check_index),
        ("demo endpoint removed", check_demo_removed),
        ("modes auto-historical", check_modes_auto),
        ("model insights", check_insights),
        ("historical always on", check_historical_always_on),
        ("scores shape + real dates", check_scores_shape_and_dates),
        ("missing artifacts error", check_missing_artifacts),
        ("no fabricated dates in source", lambda _c: check_no_fabricated_dates_in_source()),
        ("safe errors", check_safe_errors),
    ]

    failures: list[str] = []
    for name, fn in checks:
        try:
            fn(client)
            print(f"ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")

    print(json.dumps({"checks": len(checks), "failures": len(failures)}))
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nall Signal Monitor backend checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
