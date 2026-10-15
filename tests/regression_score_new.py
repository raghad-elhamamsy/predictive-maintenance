"""
Regression test for the shipped scoring path (``src.scoring``).

Plain assert-based script -- no pytest required. Run from the repository root:

    python tests/regression_score_new.py

It verifies, for **both** supervised bundles produced by notebook 05:

  1. bundle completeness (every artifact the standalone scorer needs);
  2. preprocessing reproduces ``data/interim/channel_scaled.csv`` (NaN pattern and
     values);
  3. all model features are rebuilt in training order and match the stored
     ``data/processed/features_labels.csv`` within numerical tolerance;
  4. a healthy 2032 row does not alarm and a pre-failure 2033 row does;
  5. missing core sensors, missing non-core sensors, insufficient history,
     shutdown days and restart days are all handled safely (never a spurious
     score or alarm).

Nothing here prints raw system readings: only counts, feature names and model
probabilities appear in the output. If the private artifacts are absent (for
example in a fresh clone that has not run the notebooks), the whole test skips
with a clear explanation instead of failing.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve()
ROOT = None
for candidate in [HERE.parent, *HERE.parents]:
    if (candidate / "data" / "processed" / "label_config.json").is_file():
        ROOT = candidate
        break

REQUIRED = {
    "models/05_logreg_early_warning.joblib": "logistic-regression bundle",
    "models/05_tree_early_warning.joblib": "tree bundle",
    "data/processed/features_labels.csv": "stored features/labels table",
    "data/interim/channel_clean.csv": "cleaned sensor table",
    "data/interim/channel_scaled.csv": "scaled sensor table",
    "data/processed/label_config.json": "label configuration",
}


def _skip(reason: str) -> int:
    print("SKIP regression_score_new.py:", reason)
    print("     run notebooks 01-08 from the repository root first, then retry.")
    return 0


def main() -> int:
    if ROOT is None:
        return _skip("could not locate data/processed/label_config.json")

    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        return _skip("missing artifacts: " + ", ".join(missing))

    sys.path.insert(0, str(ROOT))
    from src.scoring import (
        assert_bundle_complete, preprocess_sensors, compute_features, score_new,
        bundle_core,
    )

    cfg = json.loads((ROOT / "data" / "processed" / "label_config.json").read_text())
    windows = tuple(cfg.get("windows", (7, 14, 30)))

    clean = pd.read_csv(ROOT / "data/interim/channel_clean.csv", index_col=0, parse_dates=True)
    scaled_ref = pd.read_csv(ROOT / "data/interim/channel_scaled.csv", index_col=0, parse_dates=True)
    table = pd.read_csv(ROOT / "data/processed/features_labels.csv", index_col=0, parse_dates=True)
    assert clean.index.equals(scaled_ref.index) and clean.index.equals(table.index)
    assert (clean["is_shutdown"].astype(bool).any() and clean["is_restart"].astype(bool).any()), \
        "fixture: expected shutdown and restart rows"

    usable = table["usable"].astype(bool)
    base_day = clean.index[usable & (clean.index.year == 2032)][0]
    prefail_date = table.index[table["label_valid"].astype(bool)][-1]
    failures: list[str] = []

    for model_file, label in [
        ("models/05_logreg_early_warning.joblib", "logreg"),
        ("models/05_tree_early_warning.joblib", "tree"),
    ]:
        print(f"\n=== bundle: {label} ({model_file}) ===")
        bundle = joblib.load(ROOT / model_file)

        # ---- test 1: bundle integrity -------------------------------------
        try:
            assert_bundle_complete(bundle)
            core = bundle_core(bundle)
            assert len(bundle["features"]) == 87, len(bundle["features"])
            assert len(core) == 14, len(core)
            print(f"test 1 bundle complete: {len(bundle['features'])} features, "
                  f"{len(core)} core sensors, horizon {bundle['horizon']}, "
                  f"threshold {bundle['threshold']}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label} test 1 (bundle complete): {exc}")
            continue

        sensor_scaler = bundle["sensor_scaler"]
        scale_cols = list(sensor_scaler["columns"])
        assert scale_cols == list(sensor_scaler["scaler"].feature_names_in_)

        # ---- test 2: preprocessing parity --------------------------------
        try:
            repro = preprocess_sensors(clean[scale_cols], sensor_scaler).reindex(
                columns=scaled_ref.columns)
            nan_same = (np.isnan(repro.to_numpy()) == np.isnan(scaled_ref.to_numpy())).all()
            fin = ~np.isnan(scaled_ref.to_numpy())
            val_same = bool(np.allclose(repro.to_numpy(dtype=float)[fin],
                                        scaled_ref.to_numpy(dtype=float)[fin], atol=1e-9))
            assert nan_same and val_same, f"NaN parity {nan_same}, value parity {val_same}"
            print(f"test 2 preprocess == channel_scaled.csv: NaN pattern {nan_same}, values {val_same}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label} test 2 (preprocess parity): {exc}")

        # ---- test 3: feature parity --------------------------------------
        scaled_win = preprocess_sensors(clean[scale_cols], sensor_scaler)
        scored = score_new(clean[scale_cols], table["segment_id"], bundle, windows=windows)
        try:
            rebuilt = compute_features(scaled_win, table["segment_id"], bundle, windows=windows)
            stored = table[list(bundle["features"])]
            mask = stored.notna().all(axis=1) & rebuilt.notna().all(axis=1)
            max_abs = float(np.abs(stored.loc[mask].to_numpy(dtype=float)
                                   - rebuilt.loc[mask].to_numpy(dtype=float)).max())
            assert list(rebuilt.columns) == list(bundle["features"]), "feature order changed"
            assert max_abs < 1e-8, f"max abs feature diff {max_abs}"
            print(f"test 3 raw-input parity across all {len(bundle['features'])} features "
                  f"on {int(mask.sum())} rows: max abs diff {max_abs:.2e}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label} test 3 (feature parity): {exc}")

        # ---- test 4: healthy vs pre-failure behaviour --------------------
        try:
            healthy_alarm = bool(scored.loc[base_day, "alarm"])
            prefail_alarm = bool(scored.loc[prefail_date, "alarm"])
            assert not healthy_alarm, "healthy 2032 row should not alarm"
            assert prefail_alarm, "pre-failure 2033 row should alarm"
            print(f"test 4 behaviour: healthy 2032 alarm={healthy_alarm}, "
                  f"pre-failure alarm={prefail_alarm}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label} test 4 (healthy/pre-failure): {exc}")

        # ---- test 5: safe handling of incomplete rows --------------------
        try:
            ref = scored.loc[base_day]

            core0 = core[0]
            miss_core = clean.copy()
            miss_core.loc[base_day, core0] = np.nan
            s_core = score_new(miss_core[scale_cols], table["segment_id"], bundle,
                               windows=windows).loc[base_day]
            assert pd.isna(s_core["probability"]) and not bool(s_core["alarm"]), \
                f"missing core sensor {core0} was scored instead of NaN"

            non_core = next(c for c in scale_cols if c not in core
                            and not c.endswith("_log") and c in clean.columns)
            miss_nc = clean.copy()
            miss_nc.loc[base_day, non_core] = np.nan
            s_nc = score_new(miss_nc[scale_cols], table["segment_id"], bundle,
                             windows=windows).loc[base_day]
            assert abs(s_nc["probability"] - scored.loc[base_day, "probability"]) < 1e-9, \
                f"missing non-core sensor {non_core} changed the score"

            first_days = table.index[table["day_in_segment"].fillna(-1) == 0]
            assert len(first_days) >= 1, "fixture: expected segment-start days"
            s_first = scored.loc[first_days]
            assert s_first["probability"].isna().all(), \
                "segment-start rows should have insufficient history"

            shut = scored.loc[clean.index[clean["is_shutdown"].astype(bool)]]
            restart = scored.loc[clean.index[clean["is_restart"].astype(bool)]]
            assert shut["probability"].isna().all(), "shutdown rows must not be scored"
            assert restart["probability"].isna().all(), "restart rows must not be scored"
            print(f"test 5 safe gating: missing core -> NaN (no alarm); "
                  f"missing non-core unchanged ({non_core}); "
                  f"{len(first_days)} segment-start rows -> insufficient history (NaN); "
                  f"{len(shut)} shutdown + {len(restart)} restart rows unscored")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label} test 5 (safe gating): {exc}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nall regression checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())