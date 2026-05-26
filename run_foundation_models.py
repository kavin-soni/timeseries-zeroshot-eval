"""
Zero-shot evaluation of foundation models: TimesFM 2.5, Chronos.

Usage (on Colab GPU):
  python run_foundation_models.py --model timesfm25 --dataset traffic
  python run_foundation_models.py --model chronos   --dataset etth1
  python run_foundation_models.py --model timesfm25          # all four datasets

Evaluation protocol matches run_patchtst_all.py / run_patchtst_etth1.py:
  - Same train/test CSV splits and per-series grouping
  - Same test window: test_series[:horizon] per series
  - Same per-series MASE averaging (mean over series, NaN dropped)
  - Same seasonal periods: m=24 (Traffic, ETTh1)  m=7 (Exchange, M4)
  - Same clamps: traffic ≥ 0, exchange ≥ 1e-5
  Foundation models receive raw (original-scale) context — they perform
  their own internal normalization. MASE denominator uses raw train history,
  consistent with rec['train_hist'] in the PatchTST evaluation.

Output: results/foundation_models_results.csv (same columns as patchtst_results.csv,
        plus a leading 'model' column to distinguish runs).

---------------------------------------------------------------------------
TimesFM 2.5 — confirmed working environment & required fixes (2025-05)
---------------------------------------------------------------------------
Environment:
  torch             2.4.1+cu121
  huggingface_hub   1.15.0
  timesfm           cloned from github (reports 2.0.0, contains 2.5 code)
  GPU               NVIDIA T4 on Google Colab

FIX 1 — huggingface_hub kwargs conflict
  huggingface_hub ≥ 1.x passes extra kwargs (proxies, resume_download,
  token, revision) into TimesFM_2p5_200M_torch.__init__(), which does not
  accept them.  Patch timesfm source before running:

    File: /content/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py
    In _from_pretrained(), before: instance = cls(config=config, **model_kwargs)
    Insert:
      import inspect
      valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {'self'}
      model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_keys}

FIX 2 — max_context in ForecastConfig must equal the dataset lookback exactly
  Passing any other value (e.g. 2048) causes the model to silently allocate
  mismatched internal buffers, producing degenerate attention weights → NaN.
  This forces load_timesfm25() to be called per-dataset, not once at startup.
  Per-dataset max_context values:
    traffic  → 168    etth1    → 96
    exchange → 96     m4       → 30

FIX 3 — compile() is required before forecast()
  Calling tfm.forecast() without a prior tfm.compile(ForecastConfig(...))
  raises: "Model is not compiled. Please call compile() first."

Note: torch_compile=True raises a compilation error on T4; use False.
---------------------------------------------------------------------------
"""

# ===========================================================================
# Colab pip installs — paste into a separate cell and run BEFORE this script
# ===========================================================================
# # TimesFM 2.5 (install from source)
# !git clone https://github.com/google-research/timesfm.git
# !pip install -e timesfm/
#
# # Chronos
# !pip install chronos-forecasting
#
# # Shared deps (may already be present on Colab)
# !pip install einops huggingface_hub
# ===========================================================================

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
sys.path.insert(0, '/content/timesfm/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.metrics import calculate_metrics

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    'traffic': {
        'train_file':  'data/traffic_train.csv',
        'test_file':   'data/traffic_test.csv',
        'lookback':    168,
        'horizon':     168,
        'seasonality': 24,   # daily cycle (hourly data)
        'clamp':       0.0,  # occupancy rates are non-negative
    },
    'etth1': {
        'train_file':  'data/etth1_train.csv',
        'test_file':   'data/etth1_test.csv',
        'lookback':    96,
        'horizon':     24,
        'seasonality': 24,   # daily cycle (hourly data)
        'clamp':       None,
    },
    'exchange': {
        'train_file':  'data/exchange_train.csv',
        'test_file':   'data/exchange_test.csv',
        'lookback':    96,
        'horizon':     96,
        'seasonality': 7,    # weekly cycle (daily data)
        'clamp':       1e-5, # exchange rates are strictly positive
    },
    'm4': {
        'train_file':  'data/m4_train.csv',
        'test_file':   'data/m4_test.csv',
        'lookback':    30,
        'horizon':     14,
        'seasonality': 7,
        'clamp':       None,
    },
}

# Number of series processed per GPU forward pass.
# Increase if GPU memory allows; decrease if OOM errors occur.
INFER_BATCH = 32

# ---------------------------------------------------------------------------
# Data loading  (mirrors run_patchtst_all.py — O(n) groupby, no O(n²) filter)
# ---------------------------------------------------------------------------
def load_dataset(cfg):
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

    df_tr = df_tr.sort_values(['series_id', 'date'])
    df_te = df_te.sort_values(['series_id', 'date'])

    train_series = {sid: grp['sales'].values.astype(np.float32)
                    for sid, grp in df_tr.groupby('series_id', sort=False)}
    test_series  = {sid: grp['sales'].values.astype(np.float32)
                    for sid, grp in df_te.groupby('series_id', sort=False)}

    series_ids = sorted(train_series.keys())
    print(f"  Found {len(series_ids)} series.", flush=True)
    del df_tr, df_te
    return series_ids, train_series, test_series

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_timesfm25(lookback, horizon):
    import timesfm

    # FIX 2: max_context must equal the dataset lookback exactly.
    #        Any other value allocates mismatched buffers → silent NaN.
    #        This is why the function takes `lookback` and is called
    #        per-dataset rather than once at startup.
    print(f"  Loading TimesFM 2.5 "
          f"(max_context={lookback}, max_horizon=256)...", flush=True)

    # FIX 1: huggingface_hub passes extra kwargs that __init__ rejects;
    #        patch timesfm_2p5_torch.py before running (see module docstring).
    # FIX 3: torch_compile=True raises a compilation error on T4; use False.
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
    )

    # FIX 3: compile() with a ForecastConfig is required before forecast().
    #        normalize_inputs=True z-scores the context internally; without it,
    #        large-magnitude inputs (e.g. Traffic counts) overflow attention → NaN.
    tfm.compile(timesfm.ForecastConfig(
        max_context=lookback,   # FIX 2: must match dataset lookback exactly
        max_horizon=256,
        normalize_inputs=True,
    ))

    print("  TimesFM 2.5 ready.", flush=True)
    return tfm


def load_timesfm20(lookback, horizon):
    """
    TimesFM 2.0 (google/timesfm-1.0-200m-pytorch) requires the
    legacy pip package which only supports Python 3.10.
    Colab currently runs Python 3.12, making this model
    uninstallable in the standard environment.

    TimesFM 2.0 results in the paper were produced on Python 3.10.
    To reproduce:
      1. Use a Python 3.10 environment
      2. pip install timesfm (legacy, NOT git clone)
      3. python run_foundation_models.py --model timesfm20

    Do NOT run timesfm20 in the same session as timesfm25 or chronos.
    """
    try:
        import timesfm
        if not hasattr(timesfm, 'TimesFm'):
            raise RuntimeError(
                "TimesFM 2.0 requires Python 3.10 and the legacy pip "
                "package. Current environment is incompatible. "
                "See load_timesfm20() docstring for instructions."
            )
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=horizon,
                context_len=lookback,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
            ),
        )
        tfm.initialize()
        return tfm
    except ImportError:
        raise RuntimeError(
            "timesfm package not found. See load_timesfm20() "
            "docstring for installation instructions."
        )


def load_chronos():
    from chronos import ChronosPipeline
    print("  Loading Chronos T5-Base (amazon/chronos-t5-base)...", flush=True)
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    print("  Chronos ready.", flush=True)
    return pipeline


# ---------------------------------------------------------------------------
# Batch inference helpers
# Each function takes a list of 1-D np.float32 arrays (same length = lookback)
# and returns an np.ndarray of shape (n_series, horizon).
# ---------------------------------------------------------------------------
def predict_batch_timesfm(tfm, contexts, horizon):
    # Confirmed working call signature for TimesFM_2p5_200M_torch.
    # `horizon` is passed explicitly here; the old `freq` list argument
    # belonged to the legacy TimesFm() API and is not used in 2.5.
    point_forecast, _ = tfm.forecast(
        horizon=horizon,
        inputs=contexts,
    )
    return np.asarray(point_forecast)


def predict_batch_chronos(pipeline, contexts, horizon):
    # pipeline.predict accepts a (batch, seq_len) tensor of equal-length contexts.
    context_tensor = torch.tensor(np.stack(contexts))  # (n, lookback)
    forecast = pipeline.predict(context_tensor, prediction_length=horizon)
    # forecast: (n, num_samples, horizon) — take median across samples
    return forecast.median(dim=1).values.numpy()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_dataset(model_name, dataset_name, model_obj, no_save=False):
    cfg = DATASET_CONFIGS[dataset_name]
    L   = cfg['lookback']
    H   = cfg['horizon']
    m   = cfg['seasonality']

    print(f"\n{'='*60}")
    print(f"  Model    : {model_name}")
    print(f"  Dataset  : {dataset_name.upper()}")
    print(f"  Lookback : {L}   Horizon : {H}   m={m}")
    print(f"{'='*60}", flush=True)

    series_ids, train_series, test_series = load_dataset(cfg)

    # Build test records.  Context = last L raw training points (no external
    # normalization — foundation models normalise internally).
    test_records = []
    skipped = 0
    for sid in series_ids:
        tr = train_series[sid]
        te = test_series[sid]
        if len(tr) < L:
            skipped += 1
            continue
        if len(te) < H:
            skipped += 1
            continue
        test_records.append({
            'sid':        sid,
            'context':    tr[-L:].astype(np.float32),
            'y_true':     te[:H].astype(np.float32),
            'train_hist': tr,           # raw, for MASE denominator
        })
    if skipped:
        print(f"  Skipped {skipped} series (too short).", flush=True)

    n_series = len(test_records)
    print(f"  Evaluating {n_series} series | batch size = {INFER_BATCH}", flush=True)

    per_series           = []
    series_done          = 0
    first_batch_reported = False
    t_start              = time.time()

    for batch_start in range(0, n_series, INFER_BATCH):
        batch    = test_records[batch_start : batch_start + INFER_BATCH]
        contexts = [rec['context'] for rec in batch]

        # --- Forward pass ---
        if model_name == 'timesfm25':
            preds = predict_batch_timesfm(model_obj, contexts, H)
        elif model_name == 'timesfm20':
            point_forecast, _ = model_obj.forecast(
                inputs=contexts,
                freq=[0] * len(contexts),
            )
            preds = np.asarray(point_forecast)
        elif model_name == 'chronos':
            preds = predict_batch_chronos(model_obj, contexts, H)

        # --- Per-series metrics ---
        clamp = cfg['clamp']
        for i, rec in enumerate(batch):
            pred = preds[i].astype(np.float32)
            if clamp is not None:
                pred = np.maximum(clamp, pred)
            met = calculate_metrics(rec['y_true'], pred, rec['train_hist'],
                                    seasonality=m)
            met['series_id'] = rec['sid']
            per_series.append(met)

        # --- Progress & timing ---
        prev_done   = series_done
        series_done += len(batch)

        if not first_batch_reported:
            first_batch_reported = True
            elapsed   = time.time() - t_start
            est_total = elapsed / series_done * n_series
            print(f"  [first batch done] {series_done}/{n_series} series | "
                  f"{elapsed:.1f}s elapsed → est. total {est_total/60:.1f} min",
                  flush=True)
        else:
            # Print whenever we cross a 100-series milestone
            prev_milestone = prev_done // 100
            curr_milestone = series_done // 100
            if curr_milestone > prev_milestone or series_done == n_series:
                elapsed = time.time() - t_start
                pct     = series_done / n_series * 100
                print(f"  [{series_done}/{n_series} series | {pct:.0f}%] "
                      f"{elapsed:.1f}s elapsed", flush=True)

    # --- Aggregate ---
    results_df   = pd.DataFrame(per_series).set_index('series_id')
    mean_metrics = results_df.dropna(subset=['MASE']).mean()

    print(f"\n--- {model_name} | {dataset_name.upper()} ---")
    print(f"  RMSE  : {mean_metrics['RMSE']:.4f}")
    print(f"  sMAPE : {mean_metrics['sMAPE']:.4f}")
    print(f"  MASE  : {mean_metrics['MASE']:.4f}")
    print(f"  MAE   : {mean_metrics['MAE']:.4f}")
    print(f"  Bias  : {mean_metrics['Bias']:.2f}%", flush=True)

    if not no_save:
        os.makedirs('results', exist_ok=True)
        csv_path = 'results/foundation_models_results.csv'
        row = pd.DataFrame([{
            'model':    model_name,
            'dataset':  dataset_name,
            'lookback': L,
            'horizon':  H,
            'RMSE':     round(float(mean_metrics['RMSE']),  4),
            'sMAPE':    round(float(mean_metrics['sMAPE']), 4),
            'MASE':     round(float(mean_metrics['MASE']),  4),
            'MAE':      round(float(mean_metrics['MAE']),   4),
            'Bias':     round(float(mean_metrics['Bias']),  2),
        }])
        write_header = not os.path.exists(csv_path)
        row.to_csv(csv_path, mode='a', index=False, header=write_header)
        print(f"  Appended to {csv_path}", flush=True)

        # Per-series results for complexity router
        _ps_path = 'results/per_series_results.csv'
        _ps_rows = [
            {'model': model_name, 'dataset': dataset_name,
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
            print(f"  Per-series: {len(_ps_rows)} rows → {_ps_path}", flush=True)
    else:
        print("  (--no-save: result not written)", flush=True)

    return mean_metrics

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(model_name, all_results):
    """
    Print a MASE / sMAPE / RMSE table across all evaluated datasets,
    mirroring the format used in run_patchtst_all.py's end-of-run summary.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SUMMARY — {model_name}")
    print(sep)
    print(f"  {'Dataset':<12}  {'MASE':>8}  {'sMAPE':>8}  {'RMSE':>10}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}")
    for ds, m in all_results.items():
        print(f"  {ds:<12}  {float(m['MASE']):>8.4f}  "
              f"{float(m['sMAPE']):>8.4f}  {float(m['RMSE']):>10.4f}")
    print(sep)
    print(flush=True)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Zero-shot foundation model evaluation')
    parser.add_argument('--model',
                        choices=['timesfm25', 'timesfm20', 'chronos'],
                        required=True,
                        help='Which foundation model to evaluate')
    parser.add_argument('--dataset',
                        choices=['traffic', 'etth1', 'exchange', 'm4'],
                        default=None,
                        help='Dataset to run (default: all four in order)')
    parser.add_argument('--no-save',
                        action='store_true',
                        help='Print results only; do not append to CSV')
    args = parser.parse_args()

    datasets = ([args.dataset] if args.dataset
                else ['traffic', 'etth1', 'exchange', 'm4'])

    # ------------------------------------------------------------------
    # Model loading strategy
    #
    # TimesFM  — ForecastConfig.max_context must equal the dataset lookback
    #            exactly (FIX 2: any mismatch → silent NaN). Loaded per dataset;
    #            HuggingFace caches weights after the first download so
    #            subsequent loads are fast (disk read only, no re-download).
    #
    # Chronos  — prediction_length is passed at pipeline.predict() time,
    #            so one load covers all datasets.
    # ------------------------------------------------------------------
    print(f"\nLoading model: {args.model}", flush=True)
    if args.model == 'chronos':
        shared_model = load_chronos()
    else:
        shared_model = None  # TimesFM variants are loaded per dataset (FIX 2)

    all_results = {}
    for ds in datasets:
        cfg  = DATASET_CONFIGS[ds]
        L, H = cfg['lookback'], cfg['horizon']
        if args.model == 'timesfm25':
            model_obj = load_timesfm25(L, H)
        elif args.model == 'timesfm20':
            model_obj = load_timesfm20(L, H)
        else:
            model_obj = shared_model
        all_results[ds] = run_dataset(
            args.model, ds, model_obj, no_save=args.no_save)

    if len(all_results) > 1:
        print_summary(args.model, all_results)
