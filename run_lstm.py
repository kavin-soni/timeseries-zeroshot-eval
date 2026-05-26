"""
LSTM evaluation on Traffic, ETTh1, Exchange, and M4 Daily.

Uses a global univariate LSTM with a linear head for direct multi-step
prediction. All series from a dataset are pooled into one model, consistent
with run_dlinear_all.py and run_patchtst_all.py.

Architecture:
  Input  → LSTM(input_size=1, hidden_size=64, num_layers=2)
          → last hidden state → Linear(64, horizon)

Training:
  - Adam, MSE loss, early stopping on validation loss
  - Same normalization as DLinear (z-score or log1p)

Usage:
  python run_lstm.py                       # all four datasets
  python run_lstm.py --dataset traffic
  python run_lstm.py --no-save             # print only

Output:
  results/lstm_results.csv       (aggregate, appended)
  results/per_series_results.csv (per-series MASE/RMSE/sMAPE, appended)
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from src.utils.metrics import calculate_metrics

# ---------------------------------------------------------------------------
# Dataset configurations (mirrors run_dlinear_all.py)
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    'traffic': {
        'train_file':  'data/traffic_train.csv',
        'test_file':   'data/traffic_test.csv',
        'value_col':   'value',
        'lookback':    168,
        'horizon':     168,
        'seasonality': 24,
        'norm':        'log1p',
        'clamp':       0,
        'stride':      48,
        'epochs':      20,
        'patience':    3,
        'batch_size':  64,
        'lr':          1e-3,
        'val_ratio':   0.2,
    },
    'etth1': {
        'train_file':  'data/etth1_train.csv',
        'test_file':   'data/etth1_test.csv',
        'value_col':   'value',
        'lookback':    96,
        'horizon':     24,
        'seasonality': 24,
        'norm':        'zscore',
        'clamp':       None,
        'stride':      1,
        'epochs':      30,
        'patience':    5,
        'batch_size':  128,
        'lr':          1e-3,
        'val_ratio':   0.2,
    },
    'exchange': {
        'train_file':  'data/exchange_train.csv',
        'test_file':   'data/exchange_test.csv',
        'value_col':   'value',
        'lookback':    96,
        'horizon':     96,
        'seasonality': 7,
        'norm':        'zscore',
        'clamp':       1e-5,
        'stride':      1,
        'epochs':      30,
        'patience':    5,
        'batch_size':  128,
        'lr':          1e-3,
        'val_ratio':   0.2,
    },
    'm4': {
        'train_file':  'data/m4_train.csv',
        'test_file':   'data/m4_test.csv',
        'value_col':   'value',
        'lookback':    30,
        'horizon':     14,
        'seasonality': 7,
        'norm':        'zscore',
        'clamp':       None,
        'stride':      1,
        'epochs':      5,      # CPU-friendly cap (many series, many windows)
        'patience':    2,
        'batch_size':  256,
        'lr':          1e-3,
        'val_ratio':   0.2,
    },
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    """Univariate LSTM → direct multi-step prediction."""

    def __init__(self, lookback, horizon, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lookback     = lookback
        self.horizon      = horizon
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, lookback)
        x = x.unsqueeze(-1)                  # (batch, lookback, 1)
        _, (h, _) = self.lstm(x)             # h: (num_layers, batch, hidden)
        out = self.head(h[-1])               # (batch, horizon)
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def fit_scaler(vals, method):
    if method == 'log1p':
        return {'method': 'log1p'}
    mean = float(vals.mean())
    std  = float(vals.std())
    return {'method': 'zscore', 'mean': mean, 'std': max(std, 1e-8)}


def normalise(vals, params):
    if params['method'] == 'log1p':
        return np.log1p(np.maximum(0, vals))
    return (vals - params['mean']) / params['std']


def inverse(vals, params):
    if params['method'] == 'log1p':
        return np.expm1(vals)
    return vals * params['std'] + params['mean']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(cfg):
    df_tr = pd.read_csv(cfg['train_file'], low_memory=False)
    df_te = pd.read_csv(cfg['test_file'],  low_memory=False)

    for df in (df_tr, df_te):
        df['series_id'] = df['series_id'].astype(str)
        if 'value' in df.columns:
            df.rename(columns={'value': 'sales'}, inplace=True)

    df_tr = df_tr.sort_values(['series_id', 'date'])
    df_te = df_te.sort_values(['series_id', 'date'])

    train_series = {sid: grp['sales'].values.astype(np.float32)
                    for sid, grp in df_tr.groupby('series_id', sort=False)}
    test_series  = {sid: grp['sales'].values.astype(np.float32)
                    for sid, grp in df_te.groupby('series_id', sort=False)}

    series_ids = sorted(train_series.keys())
    print(f"  Found {len(series_ids)} series.", flush=True)
    return series_ids, train_series, test_series


# ---------------------------------------------------------------------------
# Window building
# ---------------------------------------------------------------------------
def collect_windows(vals, lookback, horizon, stride=1):
    X, Y = [], []
    n = len(vals)
    for i in range(0, n - lookback - horizon + 1, stride):
        X.append(vals[i: i + lookback])
        Y.append(vals[i + lookback: i + lookback + horizon])
    return X, Y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, patience, lr):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val  = float('inf')
    pat_count = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
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
                x, y = x.to(device), y.to(device)
                va_loss += criterion(model(x), y).item() * len(x)
        va_loss /= len(val_loader.dataset)

        elapsed = time.time() - t0
        if epoch == 1:
            est = elapsed * (epochs - 1)
            print(f"  Epoch {epoch:02d} | train={tr_loss:.6f} | val={va_loss:.6f} | "
                  f"{elapsed:.1f}s  →  est. {est/60:.1f} min remaining", flush=True)
        else:
            print(f"  Epoch {epoch:02d} | train={tr_loss:.6f} | val={va_loss:.6f} | "
                  f"{elapsed:.1f}s", flush=True)

        if va_loss < best_val:
            best_val  = va_loss
            pat_count = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"  Early stopping at epoch {epoch} (best val={best_val:.6f})",
                      flush=True)
                break

    model.load_state_dict(best_state)
    model = model.cpu()
    return model


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_dataset(name, no_save=False):
    cfg    = DATASET_CONFIGS[name]
    L, H   = cfg['lookback'], cfg['horizon']
    stride = cfg.get('stride', 1)

    print(f"\n{'='*60}")
    print(f"  Model   : LSTM (global)")
    print(f"  Dataset : {name.upper()}")
    print(f"  Lookback: {L}   Horizon: {H}   stride={stride}")
    print(f"{'='*60}", flush=True)

    print("  Loading data...", flush=True)
    series_ids, train_series, test_series = load_dataset(cfg)

    all_X_tr, all_Y_tr = [], []
    all_X_va, all_Y_va = [], []
    test_records = []

    for sid in series_ids:
        tr = train_series[sid]
        te = test_series.get(sid, np.array([], dtype=np.float32))

        if len(tr) < L or len(te) < H:
            continue

        params  = fit_scaler(tr, cfg['norm'])
        tr_norm = normalise(tr, params).astype(np.float32)

        n       = len(tr_norm)
        val_cut = int(n * (1 - cfg['val_ratio']))

        # Training windows
        Xs, Ys = collect_windows(tr_norm[:val_cut], L, H, stride)
        all_X_tr.extend(Xs)
        all_Y_tr.extend(Ys)

        # Validation windows (allow lookback overlap)
        part = tr_norm[max(0, val_cut - L):]
        Xs, Ys = collect_windows(part, L, H, 1)
        all_X_va.extend(Xs)
        all_Y_va.extend(Ys)

        test_records.append({
            'sid':          sid,
            'context_norm': tr_norm[-L:],
            'y_true_orig':  te[:H].astype(np.float32),
            'train_hist':   tr,
            'params':       params,
        })

    X_tr = np.array(all_X_tr, dtype=np.float32)
    Y_tr = np.array(all_Y_tr, dtype=np.float32)
    X_va = np.array(all_X_va, dtype=np.float32)
    Y_va = np.array(all_Y_va, dtype=np.float32)
    print(f"  Windows — train: {len(X_tr):,}, val: {len(X_va):,}", flush=True)

    train_loader = DataLoader(WindowDataset(X_tr, Y_tr),
                              batch_size=cfg['batch_size'],
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(WindowDataset(X_va, Y_va),
                              batch_size=cfg['batch_size'],
                              shuffle=False, drop_last=False)
    del X_tr, Y_tr, X_va, Y_va

    model = LSTMForecaster(lookback=L, horizon=H)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  LSTM: {n_params:,} parameters\n", flush=True)

    print("--- Training ---", flush=True)
    model = train_model(model, train_loader, val_loader,
                        cfg['epochs'], cfg['patience'], cfg['lr'])

    # --- Evaluate ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()
    per_series = []

    for rec in test_records:
        x = torch.tensor(rec['context_norm'], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(x).squeeze(0).cpu().numpy()

        pred_orig = inverse(pred_norm, rec['params'])
        if cfg['clamp'] is not None:
            pred_orig = np.maximum(cfg['clamp'], pred_orig)

        m = calculate_metrics(rec['y_true_orig'], pred_orig,
                              rec['train_hist'], seasonality=cfg['seasonality'])
        m['series_id'] = rec['sid']
        per_series.append(m)

    results_df   = pd.DataFrame(per_series).set_index('series_id')
    mean_metrics = results_df.dropna(subset=['MASE']).mean()

    print(f"\n--- LSTM | {name.upper()} ---")
    print(f"Mean RMSE:  {mean_metrics['RMSE']:.4f}")
    print(f"Mean sMAPE: {mean_metrics['sMAPE']:.4f}")
    print(f"Mean MASE:  {mean_metrics['MASE']:.4f}")
    print(f"Mean MAE:   {mean_metrics['MAE']:.4f}")
    print(f"Mean Bias:  {mean_metrics['Bias']:.2f}%")

    if not no_save:
        os.makedirs('results', exist_ok=True)

        # Aggregate CSV
        csv_path = 'results/lstm_results.csv'
        row = pd.DataFrame([{
            'model':    'lstm',
            'dataset':  name,
            'lookback': L,
            'horizon':  H,
            'RMSE':     round(float(mean_metrics['RMSE']),  4),
            'sMAPE':    round(float(mean_metrics['sMAPE']), 4),
            'MASE':     round(float(mean_metrics['MASE']),  4),
            'MAE':      round(float(mean_metrics['MAE']),   4),
            'Bias':     round(float(mean_metrics['Bias']),  2),
        }])
        row.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
        print(f"Results appended to {csv_path}")

        # Per-series results for complexity router
        _ps_path = 'results/per_series_results.csv'
        _ps_rows = [
            {'model': 'lstm', 'dataset': name,
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
    parser = argparse.ArgumentParser(description='LSTM evaluation (global model)')
    parser.add_argument('--dataset',
                        choices=['traffic', 'etth1', 'exchange', 'm4'],
                        default=None,
                        help='Dataset to run (default: all four)')
    parser.add_argument('--no-save', action='store_true',
                        help='Print only — do not append to CSV')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ['traffic', 'etth1', 'exchange', 'm4']

    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(ds, no_save=args.no_save)

    if len(all_results) > 1:
        print("\n" + "="*60)
        print("  SUMMARY — LSTM Mean MASE")
        print("="*60)
        for ds, m in all_results.items():
            print(f"  {ds:<12} MASE={m['MASE']:.4f}")
        print()
        print("Reference baselines (combined_results.csv):")
        print("  traffic MASE=0.861 | etth1 MASE=0.836 | "
              "exchange MASE=3.417 | m4 MASE=1.211")
