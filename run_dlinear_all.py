"""
DLinear evaluation on Traffic, Exchange, and M4 Daily.

Usage:
  python run_dlinear_all.py               # all three datasets
  python run_dlinear_all.py --dataset traffic
  python run_dlinear_all.py --dataset exchange
  python run_dlinear_all.py --dataset m4

NOTE (Traffic horizon): The notebook Models_for_Other_dataset.ipynb sets
FORECAST_HORIZON=168 for Traffic and its cell outputs match Table 2 exactly
(XGBoost MASE=0.514, LSTM MASE=0.861). The author's recollection of H=96
appears to be incorrect. This script uses H=168 to ensure DLinear is
comparable to the published baselines.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from src.models.dlinear import DLinear
from src.utils.metrics import calculate_metrics

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    'traffic': {
        'train_file':  'data/traffic_train.csv',
        'test_file':   'data/traffic_test.csv',
        'value_col':   'value',
        'lookback':    168,   # must equal seasonality (168h) so model sees full weekly cycle
        'horizon':     168,   # H=168 confirmed from notebook cell outputs (Table 2 match)
        'seasonality': 24,    # notebook uses daily (m=24) for MASE denominator, not weekly
        'norm':        'log1p',  # confirmed from TrafficGenerator: log1p only, no zscore
        'clamp':       0,     # occupancy rates are non-negative
        'stride':      48,    # skip every 2 days; 862 series × ~14K rows would yield ~12M windows at stride=1
        'epochs':      20,
        'batch_size':  32,
        'patience':    3,
        'val_ratio':   0.2,
        'kernel_size': 25,
        'lr':          1e-3,
    },
    'exchange': {
        'train_file':  'data/exchange_train.csv',
        'test_file':   'data/exchange_test.csv',
        'value_col':   'value',
        'lookback':    96,
        'horizon':     96,
        'seasonality': 7,
        'norm':        'zscore',
        'clamp':       1e-5,  # exchange rates are strictly positive
        'stride':      1,
        'epochs':      20,
        'batch_size':  32,
        'patience':    3,
        'val_ratio':   0.2,
        'kernel_size': 25,
        'lr':          1e-3,
    },
    'm4': {
        'train_file':  'data/m4_train.csv',
        'test_file':   'data/m4_test.csv',
        'value_col':   'sales',
        'lookback':    30,
        'horizon':     14,
        'seasonality': 7,
        'norm':        'zscore',
        'stride':      7,     # 4227 series × ~2.4K rows would yield ~10M windows at stride=1
        'epochs':      20,
        'batch_size':  256,   # larger batch for the many small series
        'patience':    3,
        'val_ratio':   0.2,
        'kernel_size': 25,
        'lr':          1e-3,
    },
}

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def fit_and_normalise(series: np.ndarray, method: str):
    """Returns (normalised_series, scaler_params)."""
    if method == 'log1p':
        normed = np.log1p(np.maximum(series, 0.0))
        return normed, None   # no per-series params needed
    else:  # zscore
        mean = series.mean()
        std  = series.std()
        std  = std if std > 1e-8 else 1.0
        return (series - mean) / std, (mean, std)

def inverse(series: np.ndarray, method: str, params):
    if method == 'log1p':
        return np.expm1(series)
    else:
        mean, std = params
        return series * std + mean

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(cfg: dict):
    vcol = cfg['value_col']

    print("  Loading train CSV...", flush=True)
    df_tr = pd.read_csv(cfg['train_file'], low_memory=False)
    print(f"  Loaded {len(df_tr):,} train rows.", flush=True)

    print("  Loading test CSV...", flush=True)
    df_te = pd.read_csv(cfg['test_file'],  low_memory=False)
    print(f"  Loaded {len(df_te):,} test rows.", flush=True)

    for df in (df_tr, df_te):
        df['series_id'] = df['series_id'].astype(str)
        # Rename value col if needed — skip expensive to_datetime, sort by string date (ISO format sorts correctly)
        if vcol != 'sales' and 'value' in df.columns:
            df.rename(columns={'value': 'sales'}, inplace=True)
        elif vcol == 'sales' and 'sales' not in df.columns and 'value' in df.columns:
            df.rename(columns={'value': 'sales'}, inplace=True)

    val_col = 'sales' if 'sales' in df_tr.columns else 'value'

    print("  Sorting and grouping...", flush=True)
    df_tr = df_tr.sort_values(['series_id', 'date'])
    df_te = df_te.sort_values(['series_id', 'date'])

    # Use groupby — O(n) instead of 4227 × O(n) individual filters
    train_series = {sid: grp[val_col].values.astype(np.float32)
                    for sid, grp in df_tr.groupby('series_id', sort=False)}
    test_series  = {sid: grp[val_col].values.astype(np.float32)
                    for sid, grp in df_te.groupby('series_id', sort=False)}

    series_ids = sorted(train_series.keys())
    print(f"  Grouped into {len(series_ids)} series.", flush=True)

    # Free raw DataFrames immediately to save RAM
    del df_tr, df_te

    return series_ids, train_series, test_series

# ---------------------------------------------------------------------------
# Sliding-window dataset — pre-allocated numpy arrays, no per-item tensor creation
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # X: (N, lookback)  Y: (N, horizon)  — float32 numpy arrays
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def collect_windows(series_ids, train_series, test_series, cfg):
    """
    Iterates over all series, normalises, splits train/val, builds windows.
    Returns pre-allocated numpy arrays and test_records list.
    Progress printed every 500 series.
    """
    L, H      = cfg['lookback'], cfg['horizon']
    stride    = cfg['stride']
    val_ratio = cfg['val_ratio']
    norm_meth = cfg['norm']

    all_X_tr, all_Y_tr = [], []
    all_X_va, all_Y_va = [], []
    test_records = []

    print(f"  Creating windows (lookback={L}, horizon={H}, stride={stride})...", flush=True)

    for i, sid in enumerate(series_ids):
        if i % 500 == 0:
            print(f"  ... series {i}/{len(series_ids)}", flush=True)

        tr = train_series[sid]
        te = test_series[sid]

        tr_norm, params = fit_and_normalise(tr, norm_meth)
        n       = len(tr_norm)
        val_cut = int(n * (1 - val_ratio))

        # Training windows
        part = tr_norm[:val_cut]
        for j in range(0, len(part) - L - H + 1, stride):
            all_X_tr.append(part[j:j+L])
            all_Y_tr.append(part[j+L:j+L+H])

        # Validation windows (with lookback overlap)
        part = tr_norm[max(0, val_cut - L):]
        for j in range(0, len(part) - L - H + 1, stride):
            all_X_va.append(part[j:j+L])
            all_Y_va.append(part[j+L:j+L+H])

        test_records.append({
            'sid':          sid,
            'context_norm': tr_norm[-L:],
            'y_true_orig':  te[:H].astype(np.float32),
            'train_hist':   tr,
            'norm_method':  norm_meth,
            'params':       params,
        })

    print(f"  Stacking arrays...", flush=True)
    X_tr = np.array(all_X_tr, dtype=np.float32)
    Y_tr = np.array(all_Y_tr, dtype=np.float32)
    X_va = np.array(all_X_va, dtype=np.float32)
    Y_va = np.array(all_Y_va, dtype=np.float32)

    print(f"  Train windows: {len(X_tr):,}  Val windows: {len(X_va):,}", flush=True)
    return X_tr, Y_tr, X_va, Y_va, test_records

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, patience, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val   = float('inf')
    pat_count  = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                va_loss += criterion(model(x), y).item() * len(x)
        va_loss /= len(val_loader.dataset)

        print(f"  Epoch {epoch:02d} | train={tr_loss:.6f} | val={va_loss:.6f}", flush=True)

        if va_loss < best_val:
            best_val   = va_loss
            pat_count  = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"  Early stopping at epoch {epoch} (best val={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    return model

# ---------------------------------------------------------------------------
# Main runner for one dataset
# ---------------------------------------------------------------------------
def run_dataset(name: str):
    cfg = DATASET_CONFIGS[name]
    print(f"\n{'='*60}")
    print(f"  Dataset : {name.upper()}")
    print(f"  Lookback: {cfg['lookback']}  Horizon: {cfg['horizon']}")
    print(f"  Norm    : {cfg['norm']}  Stride: {cfg['stride']}")
    print(f"{'='*60}")

    series_ids, train_series, test_series = load_dataset(cfg)

    X_tr, Y_tr, X_va, Y_va, test_records = collect_windows(
        series_ids, train_series, test_series, cfg)

    # Free per-series dicts now that windows are built
    del train_series, test_series

    L, H = cfg['lookback'], cfg['horizon']

    print(f"  Building DataLoaders...", flush=True)
    train_loader = DataLoader(WindowDataset(X_tr, Y_tr), batch_size=cfg['batch_size'],
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(WindowDataset(X_va, Y_va), batch_size=cfg['batch_size'],
                              shuffle=False, drop_last=False)
    del X_tr, Y_tr, X_va, Y_va

    model    = DLinear(lookback=L, horizon=H, kernel_size=cfg['kernel_size'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DLinear: {n_params} parameters\n")

    model = train_model(model, train_loader, val_loader,
                        cfg['epochs'], cfg['patience'], cfg['lr'])

    # Evaluate — per-series MASE, then mean
    model.eval()
    per_series = []

    for rec in test_records:
        x = torch.tensor(rec['context_norm'], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_norm = model(x).squeeze(0).numpy()

        pred_orig = inverse(pred_norm, rec['norm_method'], rec['params'])
        if cfg.get('clamp') is not None:
            pred_orig = np.maximum(cfg['clamp'], pred_orig)

        if len(rec['y_true_orig']) < H:
            continue   # skip series with insufficient test data

        m = calculate_metrics(rec['y_true_orig'], pred_orig,
                              rec['train_hist'], seasonality=cfg['seasonality'])
        m['series_id'] = rec['sid']
        per_series.append(m)

    results_df   = pd.DataFrame(per_series).set_index('series_id')
    mean_metrics = results_df.dropna(subset=['MASE']).mean()

    print(f"\n--- {name.upper()} Results (per-series mean) ---")
    print(f"Mean RMSE:  {mean_metrics['RMSE']:.4f}")
    print(f"Mean sMAPE: {mean_metrics['sMAPE']:.4f}")
    print(f"Mean MASE:  {mean_metrics['MASE']:.4f}")
    print(f"Mean MAE:   {mean_metrics['MAE']:.4f}")
    print(f"Mean Bias:  {mean_metrics['Bias']:.2f}%")

    # Append this dataset's row to results/dlinear_results.csv (skipped if --no-save)
    if not cfg.get('_no_save'):
        import os
        os.makedirs('results', exist_ok=True)
        csv_path = cfg.get('_out_csv') or 'results/dlinear_results.csv'
        row = pd.DataFrame([{
            'model':    'dlinear',
            'dataset':  name,
            'lookback': cfg['lookback'],
            'horizon':  cfg['horizon'],
            'RMSE':     round(float(mean_metrics['RMSE']),  4),
            'sMAPE':    round(float(mean_metrics['sMAPE']), 4),
            'MASE':     round(float(mean_metrics['MASE']),  4),
            'MAE':      round(float(mean_metrics['MAE']),   4),
            'Bias':     round(float(mean_metrics['Bias']),  2),
        }])
        write_header = not os.path.exists(csv_path)
        row.to_csv(csv_path, mode='a', index=False, header=write_header)
        print(f"Results appended to {csv_path}")

        # Per-series results for complexity router
        _ps_path = 'results/per_series_results.csv'
        _ps_rows = [
            {'model': 'dlinear', 'dataset': name,
             'series_id': d['series_id'],
             'MASE':  round(float(d['MASE']),  6),
             'RMSE':  round(float(d['RMSE']),  6),
             'sMAPE': round(float(d['sMAPE']), 6)}
            for d in per_series
            if not np.isnan(float(d.get('MASE', float('nan'))))
        ]
        if _ps_rows:
            pd.DataFrame(_ps_rows).to_csv(
                _ps_path, mode='a', index=False,
                header=not os.path.exists(_ps_path))
            print(f"  Per-series: {len(_ps_rows)} rows → {_ps_path}")
    else:
        print("(--no-save: result not written to CSV)")

    return mean_metrics

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['traffic', 'exchange', 'm4'],
                        default=None, help='Run a single dataset (default: all)')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Override horizon for the selected dataset')
    parser.add_argument('--no-save', action='store_true',
                        help='Print results only — do not append to dlinear_results.csv')
    parser.add_argument('--out-csv', type=str, default=None,
                        help='Override output CSV path (default: results/dlinear_results.csv)')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ['traffic', 'exchange', 'm4']

    if args.horizon and args.dataset:
        DATASET_CONFIGS[args.dataset]['horizon'] = args.horizon
    if args.no_save:
        for cfg in DATASET_CONFIGS.values():
            cfg['_no_save'] = True
    if args.out_csv:
        for cfg in DATASET_CONFIGS.values():
            cfg['_out_csv'] = args.out_csv

    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(ds)

    if len(all_results) > 1:
        print("\n" + "="*60)
        print("  SUMMARY — DLinear Mean MASE")
        print("="*60)
        for ds, m in all_results.items():
            print(f"  {ds:<12} MASE={m['MASE']:.4f}")
        print()
        print("Reference baselines:")
        print("  Traffic  — XGBoost=0.514, LSTM=0.861")
        print("  Exchange — XGBoost=3.942, LSTM=9.677")
        print("  M4 Daily — (no prior baselines in notebook)")
