"""Minimal Streamlit prototype for the asset early-warning model.

Read-only inference: it loads the trained bundle and scores an uploaded,
*cleaned* (unscaled) sensor window using the canonical ``src.scoring`` API.
It never retrains, never writes to the repository, and never reads the raw
system workbook.

Run from the repository root::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring import assert_bundle_complete, bundle_core, score_new  # noqa: E402

MODEL_PATH = ROOT / "models" / "05_logreg_early_warning.joblib"
MIN_HISTORY_ROWS = 15
RECOMMENDED_HISTORY_ROWS = 30
DATE_HINTS = ("date", "time", "stamp", "datetime", "timestamp")


@st.cache_resource(show_spinner=False)
def load_bundle(path_str: str):
    """Load and validate the model bundle once per session."""
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run notebook 05 to create it, or copy "
            "the 05_*.joblib artifact into the models/ directory."
        )
    bundle = joblib.load(path)
    assert_bundle_complete(bundle)
    return bundle


def _find_datetime_column(df: pd.DataFrame) -> str:
    """Pick the column that holds timestamps (by name hint, else the first column)."""
    if df.shape[1] == 0:
        raise ValueError("The uploaded CSV has no columns.")
    ordered = list(df.columns)
    hinted = [c for c in ordered if any(h in str(c).lower() for h in DATE_HINTS)]
    for col in [ordered[0], *hinted]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return col
    raise ValueError(
        "Could not find a timestamp column. Include a date/time column (for example "
        "named 'date', or the first column) with parseable timestamps."
    )


def read_and_validate(file_obj, bundle) -> tuple[pd.DataFrame, list[str]]:
    """Read an uploaded CSV into a cleaned, chronological, validated window."""
    try:
        df = pd.read_csv(file_obj)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read the CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    dt_col = _find_datetime_column(df)
    stamps = pd.to_datetime(df[dt_col], errors="coerce")
    if stamps.isna().any():
        raise ValueError(
            f"{int(stamps.isna().sum())} timestamp value(s) in '{dt_col}' could not be "
            "parsed. Fix or remove them before uploading."
        )
    window = df.drop(columns=[dt_col]).copy()
    window.index = pd.DatetimeIndex(stamps)
    window = window.loc[:, [c for c in window.columns if not str(c).startswith("Unnamed")]]

    notes: list[str] = []
    if not window.index.is_monotonic_increasing:
        window = window.sort_index()
        notes.append("Rows were not in chronological order and were reordered.")
    if window.index.has_duplicates:
        notes.append("Duplicate timestamps are present; row order is used as given.")

    if len(window) < MIN_HISTORY_ROWS:
        raise ValueError(
            f"At least {MIN_HISTORY_ROWS} rows are needed to build the rolling features, "
            f"but only {len(window)} were uploaded. Provide a window with at least "
            f"{RECOMMENDED_HISTORY_ROWS} days of history."
        )

    core = bundle_core(bundle)
    missing = [c for c in core if c not in window.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} required core sensor column(s) are missing from the upload: "
            f"{missing}. The model needs every core sensor of the trained bundle."
        )

    non_numeric = [c for c in core if not pd.api.types.is_numeric_dtype(window[c])]
    if non_numeric:
        raise ValueError(
            "These core sensor columns are not numeric and were probably not cleaned "
            f"(for example source error strings remain): {non_numeric}."
        )

    return window, notes


def _unscored_reasons(window: pd.DataFrame, core: list[str]) -> list[str]:
    """Human-readable explanations for why the latest row was not scored."""
    reasons: list[str] = []
    tail = window.tail(RECOMMENDED_HISTORY_ROWS)
    na_cores = [c for c in core if tail[c].isna().any()]
    if na_cores:
        reasons.append(
            f"{len(na_cores)} core sensor column(s) contain missing values in the recent "
            f"window: {na_cores}."
        )
    if len(window) < RECOMMENDED_HISTORY_ROWS:
        reasons.append(
            f"only {len(window)} rows were uploaded; a full 30-day rolling window needs "
            f"at least {RECOMMENDED_HISTORY_ROWS}."
        )
    if not reasons:
        reasons.append(
            "not every rolling feature could be computed from the available history "
            "(insufficient history or gaps in the core sensors)."
        )
    return reasons


def main() -> None:
    st.set_page_config(page_title="asset early-warning prototype", layout="wide")
    st.title("asset early-warning prototype")
    st.caption("Experimental machine-learning prototype. Not a certified safety system.")

    st.warning(
        "**Disclaimer.** This is an experimental research prototype. An alarm is a "
        "statistical indication only and does **not** confirm a failure; a quiet result "
        "does **not** rule one out. Supervised predictive performance is **not validated** "
        "(there is a single failure event and the failure date is provisional). Do not use "
        "this for operational decisions."
    )

    try:
        bundle = load_bundle(str(MODEL_PATH))
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"The model bundle could not be loaded or is incomplete: {exc}")
        st.stop()

    core = bundle_core(bundle)
    with st.expander("Expected upload schema"):
        st.markdown(
            f"- One CSV, one row per timestamp.\n"
            f"- A timestamp column (for example `date`) or a datetime first column.\n"
            f"- All **{len(core)} core sensor columns** of the trained model; other "
            f"columns are ignored.\n"
            f"- **Cleaned, unscaled** values: source error strings removed, signal handled by "
            f"the shipped scaler.\n"
            f"- At least **{RECOMMENDED_HISTORY_ROWS}** rows of history recommended "
            f"(minimum {MIN_HISTORY_ROWS})."
        )
        st.code(", ".join(core), language="text")

    uploaded = st.file_uploader("Upload cleaned sensor CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV of cleaned sensor history to score the latest row.")
        return

    try:
        window, notes = read_and_validate(uploaded, bundle)
    except ValueError as exc:
        st.error(str(exc))
        return

    for note in notes:
        st.info(note)

    seg = pd.Series(1, index=window.index, dtype="int64")
    with st.spinner("Scoring..."):
        try:
            results = score_new(window, seg, bundle)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Scoring failed: {exc}")
            return

    latest = results.iloc[-1]
    st.subheader("Latest result")
    if pd.isna(latest["probability"]):
        st.warning(
            "The latest row is **not scored**. " + " ".join(_unscored_reasons(window, core))
        )
        st.caption(
            "An unscored row means one or more model features could not be computed from "
            "the uploaded history; no probability is shown on purpose."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Failure probability (within horizon)", f"{latest['probability']:.3f}")
        c2.metric("Alarm", "ON" if bool(latest["alarm"]) else "off")
        c3.metric("Horizon (days)", int(bundle["horizon"]))
        st.caption(
            f"Alarm threshold = {bundle['threshold']:.2f}; sustained for "
            f"{bundle['sustain_k']} day(s). This is an indication, not a diagnosis."
        )

    st.subheader("Probability timeline (scored rows only)")
    scored = results.dropna(subset=["probability"])
    if scored.empty:
        st.info("No row in this upload could be scored.")
    else:
        st.line_chart(scored["probability"])
        st.caption(f"Scored {len(scored)} of {len(results)} uploaded rows.")


if __name__ == "__main__":
    main()
