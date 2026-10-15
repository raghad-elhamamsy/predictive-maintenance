"""
Canonical feature building and scoring used by the deployment interface.

The notebooks 04-07 apply the same logic inline. This module is the importable
version, kept byte-for-byte compatible so that notebooks 07 and the audits can
reuse it. It performs:

1. ``preprocess_sensors`` -- turn raw, cleaned sensor values into the scaled
   window that notebook 03 produced (log-signal, then the RobustScaler that was
   fit on the 2032 healthy baseline and shipped inside the model bundle).
2. ``build_features`` -- notebook 04 rolling features, per operating segment.
3. ``compute_features`` -- all model features in the model's exact column order.
4. ``score_new`` -- probability of failure within the bundle's horizon, plus an
   alarm flag gated on every feature actually being present.

Nothing here reads the raw system workbook and no sensor values are required:
scoring only needs the columns listed in the bundle's ``sensor_scaler``.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_sensors(raw_window, scaler_bundle):
    """Raw, cleaned sensor values -> the scaled window notebook 03 built.

    signal enters the model as channel_099_log; every column is scaled by the
    RobustScaler fit once on the 2032 healthy baseline (never on future data).
    Missing cells and rows outside the returned column set become NaN in the
    scaled window, exactly as notebook 03 produced them.
    """
    raw = raw_window.copy()
    if 'channel_099_log' not in raw.columns:
        if 'channel_099' not in raw.columns:
            raise KeyError('raw window needs the asset sensor columns incl. channel_099')
        raw['channel_099_log'] = np.log1p(raw['channel_099'].clip(lower=0))
    cols = list(scaler_bundle['columns'])
    X = raw.reindex(columns=cols)
    arr = scaler_bundle['scaler'].transform(X)
    return pd.DataFrame(arr, columns=cols, index=raw.index)


# ---------------------------------------------------------------------------
# Features (same definitions and order as notebooks 04 and 05)
# ---------------------------------------------------------------------------
def rolling_slope(x):
    """Slope of the last valid points; NaN unless at least 7 valid points."""
    ok = ~np.isnan(x)
    if ok.sum() < 7:
        return np.nan
    t = np.arange(len(x))[ok]
    return np.polyfit(t, x[ok], 1)[0]


def build_features(scaled_window, seg, core, windows):
    """Per-segment rolling features over the scaled core sensors."""
    seg = pd.Series(seg)
    parts = []
    for sid in seg.dropna().unique():
        mask = (seg == sid).fillna(False).to_numpy(dtype=bool)
        block = scaled_window.loc[mask, core]
        out = {}
        for c in core:
            s = block[c]
            for w in windows:
                out[f'{c}__mean{w}'] = s.rolling(w, min_periods=max(3, w // 2)).mean()
            out[f'{c}__std14'] = s.rolling(14, min_periods=7).std()
            out[f'{c}__slope14'] = s.rolling(14, min_periods=7).apply(rolling_slope, raw=True)
            out[f'{c}__diff1'] = s.diff()
        parts.append(pd.DataFrame(out, index=block.index))
    return pd.concat(parts).sort_index().reindex(scaled_window.index)


def bundle_core(bundle):
    """Core sensor order recovered from the model's feature list."""
    bases = []
    for f in bundle['features']:
        if '__' in f:
            base = f.split('__')[0]
            if base not in bases:
                bases.append(base)
    return bases


def compute_features(scaled_window, seg, bundle, windows=(7, 14, 30), core=None):
    """All model features in training order, using only past information.

    ``scaled_window`` must already be the scaled table (see ``preprocess_sensors``).
    The persisted ``mahal_baseline`` covariance is used so the summary scores
    reproduce notebook 04 exactly without refitting anything.
    """
    core = core if core is not None else bundle_core(bundle)
    feats = build_features(scaled_window, seg, core, windows)
    m7 = feats[[f'{c}__mean7' for c in core]]
    feats['health_abs7'] = m7.abs().mean(axis=1, skipna=False)
    feats['health_max7'] = m7.abs().max(axis=1, skipna=False)
    feats['mahal7'] = np.nan
    mb = bundle['mahal_baseline']
    ok = m7.notna().all(axis=1).to_numpy()
    feats.loc[ok, 'mahal7'] = np.sqrt(mb['estimator'].mahalanobis(
        m7.loc[ok, mb['mean7_columns']].to_numpy(dtype=float)))
    return feats.reindex(columns=bundle['features'])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_new(raw_window, seg, bundle, windows=(7, 14, 30)):
    """Probability of failure within the bundle horizon and an alarm flag.

    Rows whose 87 features are not all available (missing core sensor, missing
    history at the start of a segment, shutdown or restart day) are *not*
    scored: the probability is NaN and the alarm is False. This mirrors the row
    gating used when the model was trained and evaluated: an incomplete row
    never produces a spurious alarm.
    """
    scaled_window = preprocess_sensors(raw_window, bundle['sensor_scaler'])
    Xnew = compute_features(scaled_window, seg, bundle, windows=windows)
    ready = Xnew.notna().all(axis=1).to_numpy()
    proba = pd.Series(np.nan, index=Xnew.index)
    proba.iloc[np.where(ready)[0]] = bundle['pipeline'].predict_proba(Xnew[ready])[:, 1]
    alarm = (proba >= bundle['threshold']).fillna(False)
    return pd.DataFrame({'probability': proba, 'alarm': alarm})


# ---------------------------------------------------------------------------
# Bundle integrity
# ---------------------------------------------------------------------------
def assert_bundle_complete(bundle):
    """All preprocessing artifacts the standalone scorer needs must be present."""
    required = {'pipeline', 'features', 'horizon', 'threshold',
                'sustain_k', 'mahal_baseline', 'sensor_scaler'}
    missing = sorted(required - set(bundle))
    assert not missing, f'bundle missing keys: {missing}'
    sc = bundle['sensor_scaler']
    assert {'scaler', 'columns', 'baseline_year', 'n_baseline_rows'} <= set(sc), \
        'sensor_scaler is incomplete'
    assert list(sc['columns']) == list(sc['scaler'].feature_names_in_), \
        'sensor_scaler column order does not match its fitted order'
    mb = bundle['mahal_baseline']
    assert {'estimator', 'mean7_columns', 'n_baseline_rows'} <= set(mb), \
        'mahal_baseline is incomplete'
    assert len(mb['mean7_columns']) == len(bundle_core(bundle)), \
        'mahal mean7 columns do not match the core sensor count'
    assert len(bundle['features']) == len(set(bundle['features'])), \
        'feature list contains duplicates'
    return True