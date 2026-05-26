"""
PatchTST evaluation on Traffic, Exchange, and M4 Daily.

Usage:
  python run_patchtst_all.py                  # all three datasets
  python run_patchtst_all.py --dataset traffic
  python run_patchtst_all.py --dataset exchange
  python run_patchtst_all.py --dataset m4
  python run_patchtst_all.py --dataset traffic --no-save   # skip CSV write

NOTE (M4 patch config): lookback=30 with patch_len=16 yields only 2 patches,
making attention meaningless. M4 uses patch_len=4, stride=2 → 14 patches.
All other hyperparameters match the ETTh1 config.

NOTE (Traffic horizon): H=168 confirmed from notebook cell outputs matching Table 2.
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from src.models.patchtst import PatchTST
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
        'horizon':     168,
        'seasonality': 24,    # notebook uses daily (m=24) for MASE denominator, not weekly
        'norm':        'log1p',  # confirmed from TrafficGenerator: log1p only, no zscore
        'clamp':       0,     # occupancy rates are non-negative
        'stride':      48,      # window stride to keep window count manageable
        'patch_len':   16,
        'patch_stride': 8,
        'epochs':      30,
        'batch_size':  32,
        'patience':    5,
        'val_ratio':   0.2,
        'd_model':     128,
        'n_heads':     16,
        'e_layers':    3,
        'd_ff':        256,
        'dropout':     0.2,
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
        'patch_len':   16,
        'patch_stride': 8,
        'epochs':      30,
        'batch_size':  128,
        'patience':    5,
        'val_ratio':   0.2,
        'd_model':     128,
        'n_heads':     16,
        'e_layers':    3,
        'd_ff':        256,
        'dropout':     0.2,
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
        'stride':      7,
        # patch_len=16 with lookback=30 → only 2 patches; use smaller patch instead
        'patch_len':   4,
        'patch_stride': 2,      # → n_patches = (30-4)//2 + 1 = 14
        'epochs':      5,       # CPU constraint: ~15 min/epoch × 1.1M windows
        'batch_size':  256,     # larger batch for many small series
        'patience':    2,
        'val_ratio':   0.2,
        'd_model':     128,
        'n_heads':     16,
        'e_layers':    3,
        'd_ff':        256,
        'dropout':     0.2,
        'lr':          1e-3,
    },
}

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def fit_and_normalise(series, method):
    if method == 'log1p':
        return np.log1p(np.maximum(series, 0.0).astype(np.float32)), None
    mean = series.mean()
    std  = series.std()
    std  = std if std > 1e-8 else 1.0
    return (series - mean) / std, (mean, std)

def inverse(series, method, params):
    if method == 'log1p':
        return np.expm1(series)
    mean, std = params
    return series * std + mean

# ---------------------------------------------------------------------------
# Data loading  (groupby — O(n), not O(n²))
# ---------------------------------------------------------------------------
def load_dataset(cfg):
    vcol = cfg['value_col']

    print("  Loading train CSV...", flush=True)
    df_tr = pd.read_csv(cfg['train_file'], low_memory=False)
    print(f"  Loaded {len(df_tr):,} train rows.", flush=True)

    print("  Loading test CSV...", flush=True)
    df_te = pd.read_csv(cfg['test_file'], low_memory=False)
    print(f"  Loaded {len(df_te):,} test rows.", flush=True)

    for df in (df_tr, df_te):
        df['series_id'] = df['series_id'].astype(str)
        if 'value' in df.columns:
            df.rename(columns={'value': 'sales'}, inplace=True)

    print("  Sorting and grouping...", flush=True)
    df_tr = df_tr.sort_values(['series_id', 'date'])
    df_te = df_te.sort_values(['series_id', 'date'])

    train_series = {sid: grp['sales'].values.astype(np.float32)
                    for sid, grp in df_tr.groupby('series_id', sort=False)}
    test_series  = {sid: grp['sales'].values.astype(np.float32)
                    for sid, grp in df_te.groupby('series_id', sort=False)}

    series_ids = sorted(train_series.keys())
    print(f"  Grouped into {len(series_ids)} series.", flush=True)
    del df_tr, df_te
    return series_ids, train_series, test_series

# ---------------------------------------------------------------------------
# Window dataset (pre-allocated numpy → single tensor conversion at init)
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def collect_windows(series_ids, train_series, test_series, cfg):
    L, H       = cfg['lookback'], cfg['horizon']
    stride     = cfg['stride']
    val_ratio  = cfg['val_ratio']
    norm_meth  = cfg['norm']

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

        part = tr_norm[:val_cut]
        for j in range(0, len(part) - L - H + 1, stride):
            all_X_tr.append(part[j:j+L])
            all_Y_tr.append(part[j+L:j+L+H])

        part = tr_norm[max(0, val_cut - L):]
        for j in range(0, len(part) - L - H + 1, stride):
            all_X_va.append(part[j:j+L])
            all_Y_va.append(part[j+L:j+L+H])

        test_records.append({
            'sid':         sid,
            'context_norm': tr_norm[-L:],
            'y_true_orig': te[:H].astype(np.float32),
            'train_hist':  tr,
            'norm_method': norm_meth,
            'params':      params,
        })

    print("  Stacking arrays...", flush=True)
    X_tr = np.array(all_X_tr, dtype=np.float32)
    Y_tr = np.array(all_Y_tr, dtype=np.float32)
    X_va = np.array(all_X_va, dtype=np.float32)
    Y_va = np.array(all_Y_va, dtype=np.float32)
    print(f"  Train windows: {len(X_tr):,}  Val windows: {len(X_va):,}", flush=True)
    return X_tr, Y_tr, X_va, Y_va, test_records

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.MSELoss()
    best_val   = float('inf')
    pat_count  = 0
    best_state = None
    n_batches  = len(train_loader)
    log_every  = max(1, min(500, n_batches // 5))  # ~5 prints per epoch, max every 500 batches

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()

        model.train()
        tr_loss   = 0.0
        batch_t0  = time.time()
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(x)

            if batch_idx % log_every == 0:
                elapsed_so_far = time.time() - batch_t0
                est_epoch_total = elapsed_so_far / batch_idx * n_batches
                print(f"    batch {batch_idx}/{n_batches} | "
                      f"loss={tr_loss / (batch_idx * cfg['batch_size']):.6f} | "
                      f"~{est_epoch_total/60:.1f} min/epoch", flush=True)

        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                va_loss += criterion(model(x), y).item() * len(x)
        va_loss /= len(val_loader.dataset)

        elapsed = time.time() - t0

        if epoch == 1:
            est_remaining = elapsed * (cfg['epochs'] - 1)
            print(f"  Epoch {epoch:02d} | train={tr_loss:.6f} | val={va_loss:.6f} | "
                  f"{elapsed:.1f}s  →  est. {est_remaining/60:.1f} min remaining "
                  f"(assuming no early stop)", flush=True)
        else:
            print(f"  Epoch {epoch:02d} | train={tr_loss:.6f} | val={va_loss:.6f} | "
                  f"{elapsed:.1f}s", flush=True)

        if va_loss < best_val:
            best_val   = va_loss
            pat_count  = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat_count += 1
            if pat_count >= cfg['patience']:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val={best_val:.6f})", flush=True)
                break

    model.load_state_dict(best_state)
    return model

# ---------------------------------------------------------------------------
# Runner for one dataset
# ---------------------------------------------------------------------------
def run_dataset(name, no_save=False, out_csv=None):
    cfg = DATASET_CONFIGS[name]

    print(f"\n{'='*60}")
    print(f"  Dataset   : {name.upper()}")
    print(f"  Lookback  : {cfg['lookback']}  Horizon: {cfg['horizon']}")
    print(f"  Norm      : {cfg['norm']}  Window stride: {cfg['stride']}")
    print(f"  patch_len : {cfg['patch_len']}  patch_stride: {cfg['patch_stride']}")
    print(f"  batch_size: {cfg['batch_size']}  epochs: {cfg['epochs']}  patience: {cfg['patience']}")
    print(f"{'='*60}", flush=True)

    series_ids, train_series, test_series = load_dataset(cfg)

    X_tr, Y_tr, X_va, Y_va, test_records = collect_windows(
        series_ids, train_series, test_series, cfg)
    del train_series, test_series

    print("  Building DataLoaders...", flush=True)
    train_loader = DataLoader(WindowDataset(X_tr, Y_tr),
                              batch_size=cfg['batch_size'], shuffle=True, drop_last=False)
    val_loader   = DataLoader(WindowDataset(X_va, Y_va),
                              batch_size=cfg['batch_size'], shuffle=False, drop_last=False)
    del X_tr, Y_tr, X_va, Y_va

    model = PatchTST(
        lookback  = cfg['lookback'],
        horizon   = cfg['horizon'],
        patch_len = cfg['patch_len'],
        stride    = cfg['patch_stride'],
        d_model   = cfg['d_model'],
        n_heads   = cfg['n_heads'],
        e_layers  = cfg['e_layers'],
        d_ff      = cfg['d_ff'],
        dropout   = cfg['dropout'],
    )
    n_params  = sum(p.numel() for p in model.parameters())
    n_patches = model.n_patches
    print(f"\n  PatchTST: {n_params:,} parameters | n_patches={n_patches}\n", flush=True)

    print("--- Training ---", flush=True)
    model = train_model(model, train_loader, val_loader, cfg)

    # Per-series evaluation
    model.eval()
    per_series = []
    for rec in test_records:
        x = torch.tensor(rec['context_norm'], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_norm = model(x).squeeze(0).numpy()
        pred_orig = inverse(pred_norm, rec['norm_method'], rec['params'])
        if cfg.get('clamp') is not None:
            pred_orig = np.maximum(cfg['clamp'], pred_orig)
        if len(rec['y_true_orig']) < cfg['horizon']:
            continue
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

    if not no_save:
        os.makedirs('results', exist_ok=True)
        csv_path = out_csv if out_csv else 'results/patchtst_results.csv'
        row = pd.DataFrame([{
            'model':    'patchtst',
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
            {'model': 'patchtst', 'dataset': name,
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
    parser.add_argument('--dataset', choices=['traffic', 'exchange', 'm4'], default=None)
    parser.add_argument('--horizon', type=int, default=None,
                        help='Override horizon for the selected dataset')
    parser.add_argument('--no-save', action='store_true',
                        help='Print only — do not append to CSV')
    parser.add_argument('--out-csv', type=str, default=None,
                        help='Override output CSV path (default: results/patchtst_results.csv)')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ['traffic', 'exchange', 'm4']

    if args.horizon and args.dataset:
        DATASET_CONFIGS[args.dataset]['horizon'] = args.horizon

    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(ds, no_save=args.no_save, out_csv=args.out_csv)

    if len(all_results) > 1:
        print("\n" + "="*60)
        print("  SUMMARY — PatchTST Mean MASE")
        print("="*60)
        for ds, m in all_results.items():
            print(f"  {ds:<12} MASE={m['MASE']:.4f}")
        print()
        print("DLinear reference:")
        print("  traffic=2.306 | exchange=3.417 | m4=1.211")
        print()
        print("Note: M4 capped at 5 epochs (patience=2) due to CPU cost (~15 min/epoch).")
