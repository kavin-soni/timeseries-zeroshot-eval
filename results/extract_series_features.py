"""
extract_series_features.py — Compute per-series features for the Complexity Router.

Processes Traffic (S=168) and M4 Daily (S=7) only.
ETTh1 (7 series) and Exchange (8 series) are too small for decile analysis.

Features:
  1. spectral_entropy    : Shannon entropy of the normalized FFT power spectrum
  2. cv                  : std / |mean|, denominator clamped at 1e-5
  3. seasonal_autocorr   : Pearson r between series and series shifted by S
  4. trend_strength      : R² of linear regression on the training series
  5. history_length      : number of training observations

Output: results/series_features.csv
  Columns: dataset, series_id, spectral_entropy, cv, seasonal_autocorr,
           trend_strength, history_length

Usage:
  cd /path/to/timeseries-zeroshot-eval
  python results/extract_series_features.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

# Resolve paths relative to the repo root (one directory above this script)
_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(_HERE)
_DATA_DIR = os.path.join(_REPO, 'data')
_OUT_PATH = os.path.join(_HERE, 'series_features.csv')

DATASETS = {
    'traffic': {
        'train_file': os.path.join(_DATA_DIR, 'traffic_train.csv'),
        'seasonality': 168,   # weekly cycle for hourly traffic data
    },
    'm4': {
        'train_file': os.path.join(_DATA_DIR, 'm4_train.csv'),
        'seasonality': 7,     # weekly cycle for M4 daily series
    },
}


# ---------------------------------------------------------------------------
# Feature functions — all operate on 1-D np.float32 arrays
# ---------------------------------------------------------------------------

def spectral_entropy(series: np.ndarray) -> float:
    """Shannon entropy of the normalized FFT power spectrum."""
    fft_vals = np.fft.rfft(series)
    power    = np.abs(fft_vals) ** 2
    total    = power.sum()
    if total < 1e-15:
        return 0.0
    p   = power / total
    p   = p[p > 0]          # avoid log(0)
    return float(-np.sum(p * np.log(p)))


def cv(series: np.ndarray) -> float:
    """Coefficient of variation: std / |mean|, denominator clamped at 1e-5."""
    denom = max(abs(float(series.mean())), 1e-5)
    return float(series.std() / denom)


def seasonal_autocorr(series: np.ndarray, s: int) -> float:
    """Pearson correlation between the series and itself shifted by s lags.
    Returns 0.0 if the series is too short (< s+1 points) or degenerate."""
    if len(series) <= s:
        return 0.0
    a = series[s:]
    b = series[:-s]
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(r) if np.isfinite(r) else 0.0


def trend_strength(series: np.ndarray) -> float:
    """R² of a linear regression of the series values against time index."""
    n = len(series)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    result = linregress(t, series.astype(np.float64))
    return float(result.rvalue ** 2)


# ---------------------------------------------------------------------------
# Load one dataset's training series
# ---------------------------------------------------------------------------
def load_train_series(train_file: str) -> dict:
    """Returns {series_id: np.ndarray(float32)} sorted by series_id."""
    df = pd.read_csv(train_file, low_memory=False)
    df['series_id'] = df['series_id'].astype(str)
    if 'value' in df.columns:
        df.rename(columns={'value': 'sales'}, inplace=True)
    df = df.sort_values(['series_id', 'date'])
    return {sid: grp['sales'].values.astype(np.float32)
            for sid, grp in df.groupby('series_id', sort=False)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows = []

    for dataset_name, cfg in DATASETS.items():
        print(f"\nProcessing {dataset_name.upper()} "
              f"(S={cfg['seasonality']})...", flush=True)

        train_series = load_train_series(cfg['train_file'])
        S = cfg['seasonality']

        for sid in sorted(train_series.keys()):
            series = train_series[sid]

            row = {
                'dataset':          dataset_name,
                'series_id':        sid,
                'spectral_entropy': round(spectral_entropy(series), 6),
                'cv':               round(cv(series), 6),
                'seasonal_autocorr':round(seasonal_autocorr(series, S), 6),
                'trend_strength':   round(trend_strength(series), 6),
                'history_length':   int(len(series)),
            }
            rows.append(row)

        print(f"  {len(train_series)} series processed.", flush=True)

    df_out = pd.DataFrame(rows, columns=[
        'dataset', 'series_id',
        'spectral_entropy', 'cv', 'seasonal_autocorr',
        'trend_strength', 'history_length',
    ])

    df_out.to_csv(_OUT_PATH, index=False)
    print(f"\nSaved {len(df_out)} rows to {_OUT_PATH}")
    print(df_out.groupby('dataset')[
        ['spectral_entropy', 'cv', 'seasonal_autocorr',
         'trend_strength', 'history_length']
    ].describe().round(4).to_string())


if __name__ == '__main__':
    main()
