"""
routing_analysis.py — Empirical validation of the Complexity Router.

Joins per-series model results with per-series features to answer:
  "For which series types do foundation models outperform specialists?"

Inputs:
  results/series_features.csv     (from extract_series_features.py)
  results/per_series_results.csv  (from all model evaluation scripts)

Outputs:
  figures/routing_feature_analysis.pdf  — 4 decile FM-win-rate plots
  figures/pareto_frontier.pdf           — Pareto frontier: hybrid MASE vs cost

Prints:
  Feature thresholds where FM win rate crosses 60%

Usage:
  cd /path/to/timeseries-zeroshot-eval
  python results/routing_analysis.py

Definitions:
  FM models       : timesfm20, timesfm25, chronos
  Specialist models: xgboost, lstm, patchtst, dlinear
  fm_wins = 1     : min FM MASE < min Specialist MASE (for that series)
  FM inference cost: 1000× a single specialist call
    → relative_cost(alpha) = 1 + alpha * 999, where alpha is the fraction
      of series routed to FM (sorted best-first by fm_advantage)
  hybrid_mase(alpha): mean MASE of the hybrid router across all series
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless rendering — no display needed
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

warnings.filterwarnings('ignore', category=FutureWarning)

_HERE    = os.path.dirname(os.path.abspath(__file__))
_REPO    = os.path.dirname(_HERE)
_FIG_DIR = os.path.join(_REPO, 'figures')

FEATURES_PATH    = os.path.join(_HERE, 'series_features.csv')
PER_SERIES_PATH  = os.path.join(_HERE, 'per_series_results.csv')

FM_MODELS          = {'timesfm25', 'chronos'}
SPECIALIST_MODELS  = {'xgboost', 'lstm', 'patchtst', 'dlinear'}

FEATURE_COLS = ['spectral_entropy', 'cv', 'seasonal_autocorr', 'trend_strength']
FEATURE_LABELS = {
    'spectral_entropy':  'Spectral Entropy',
    'cv':                'Coefficient of Variation',
    'seasonal_autocorr': 'Seasonal Autocorrelation (lag S)',
    'trend_strength':    'Trend Strength (R²)',
}

FM_COST_MULTIPLIER = 1000   # FM call costs 1000× a single specialist call
N_DECILES          = 10
WIN_RATE_THRESHOLD = 0.60   # 60 % FM win rate


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def load_data():
    if not os.path.exists(FEATURES_PATH):
        sys.exit(f"ERROR: {FEATURES_PATH} not found.\n"
                 "Run: python results/extract_series_features.py")
    if not os.path.exists(PER_SERIES_PATH):
        sys.exit(f"ERROR: {PER_SERIES_PATH} not found.\n"
                 "Run all model scripts with --save (default) first.")

    features = pd.read_csv(FEATURES_PATH)
    features['series_id'] = features['series_id'].astype(str)

    per_series = pd.read_csv(PER_SERIES_PATH)
    per_series['series_id'] = per_series['series_id'].astype(str)

    # Restrict to Traffic and M4 (datasets covered by extract_series_features.py)
    per_series = per_series[per_series['dataset'].isin(['traffic', 'm4'])].copy()

    return features, per_series


def build_routing_table(features, per_series):
    """Return a DataFrame with one row per (dataset, series_id) containing:
      - feature columns
      - best_fm_mase, best_specialist_mase
      - fm_advantage = specialist_best - fm_best  (positive → FM wins)
      - fm_wins (bool)
    """
    # Pivot so each (dataset, series_id, model) → MASE
    pivot = (per_series
             .pivot_table(index=['dataset', 'series_id'],
                          columns='model',
                          values='MASE',
                          aggfunc='first'))

    fm_cols   = [c for c in pivot.columns if c in FM_MODELS]
    spec_cols = [c for c in pivot.columns if c in SPECIALIST_MODELS]

    if not fm_cols:
        sys.exit("ERROR: No FM models found in per_series_results.csv.\n"
                 f"       Found models: {sorted(pivot.columns.tolist())}\n"
                 f"       Expected one of: {sorted(FM_MODELS)}")
    if not spec_cols:
        sys.exit("ERROR: No specialist models found in per_series_results.csv.\n"
                 f"       Found models: {sorted(pivot.columns.tolist())}\n"
                 f"       Expected one of: {sorted(SPECIALIST_MODELS)}")

    pivot['best_fm_mase']         = pivot[fm_cols].min(axis=1)
    pivot['best_specialist_mase'] = pivot[spec_cols].min(axis=1)
    pivot['fm_advantage']         = (pivot['best_specialist_mase']
                                     - pivot['best_fm_mase'])
    pivot['fm_wins']              = (pivot['fm_advantage'] > 0).astype(int)
    pivot = pivot.reset_index()

    # Join features
    merged = pivot.merge(
        features[['dataset', 'series_id'] + FEATURE_COLS],
        on=['dataset', 'series_id'],
        how='inner',
    )
    print(f"  Routing table: {len(merged)} series "
          f"({merged['fm_wins'].sum()} FM wins, "
          f"{len(merged) - merged['fm_wins'].sum()} specialist wins)")
    print(f"  FM models in data    : {fm_cols}")
    print(f"  Specialist models    : {spec_cols}")
    return merged


# ---------------------------------------------------------------------------
# Plot 1: FM win-rate by feature decile (4 subplots)
# ---------------------------------------------------------------------------

def plot_feature_analysis(routing_df, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    thresholds = {}  # feature → threshold value where win rate crosses 60%

    for ax, feat in zip(axes, FEATURE_COLS):
        col_data = routing_df[feat].dropna()
        if col_data.empty:
            ax.set_title(f"{FEATURE_LABELS[feat]}\n(no data)")
            continue

        # Assign deciles (1–10)
        routing_df['_decile'] = pd.qcut(
            routing_df[feat], q=N_DECILES, labels=False,
            duplicates='drop') + 1

        grouped = routing_df.groupby('_decile').agg(
            win_rate=('fm_wins', 'mean'),
            n=('fm_wins', 'count'),
            feat_median=(feat, 'median'),
        ).reset_index()

        ax.bar(grouped['_decile'], grouped['win_rate'] * 100,
               color='steelblue', alpha=0.75, edgecolor='white')
        ax.axhline(WIN_RATE_THRESHOLD * 100, color='red', linestyle='--',
                   linewidth=1.2, label=f'{int(WIN_RATE_THRESHOLD*100)}% threshold')
        ax.set_xlabel(f"Decile of {FEATURE_LABELS[feat]}", fontsize=9)
        ax.set_ylabel("FM Win Rate (%)", fontsize=9)
        ax.set_title(FEATURE_LABELS[feat], fontsize=10, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xticks(grouped['_decile'])
        ax.legend(fontsize=8)

        # Annotate each bar with n
        for _, r in grouped.iterrows():
            ax.text(r['_decile'], r['win_rate'] * 100 + 1.5,
                    f"n={int(r['n'])}", ha='center', va='bottom', fontsize=7)

        # Find threshold (first decile where win rate ≥ 60%)
        above = grouped[grouped['win_rate'] >= WIN_RATE_THRESHOLD]
        if not above.empty:
            thresh_row  = above.iloc[0]
            thresh_val  = thresh_row['feat_median']
            thresholds[feat] = thresh_val
            ax.axvline(thresh_row['_decile'], color='orange', linestyle=':',
                       linewidth=1.5, label=f'≥60% @ decile {int(thresh_row["_decile"])}')
            ax.legend(fontsize=8)
        else:
            thresholds[feat] = None

        # Clean up temp column
        routing_df.drop(columns=['_decile'], inplace=True)

    fig.suptitle("FM Win Rate by Feature Decile\n(Traffic + M4; FM = best of TimesFM20/25/Chronos)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    return thresholds


# ---------------------------------------------------------------------------
# Plot 2: Pareto frontier — hybrid MASE vs relative inference cost
# ---------------------------------------------------------------------------

def plot_pareto_frontier(routing_df, pdf):
    """
    Sweep alpha ∈ [0, 1]:
      alpha = fraction of series routed to FM (sorted by fm_advantage descending)
      relative_cost(alpha) = 1 + alpha * (FM_COST_MULTIPLIER - 1)
      hybrid_mase(alpha)   = mean of:
        - best_fm_mase   for FM-routed series (top-alpha by fm_advantage)
        - best_specialist_mase for the rest
    """
    df = routing_df.sort_values('fm_advantage', ascending=False).reset_index(drop=True)
    n  = len(df)

    alphas         = np.linspace(0, 1, 201)
    hybrid_mases   = []
    relative_costs = []

    pure_specialist_mase = df['best_specialist_mase'].mean()
    pure_fm_mase         = df['best_fm_mase'].mean()

    for alpha in alphas:
        k = int(round(alpha * n))   # number of series routed to FM
        mase_vals = np.concatenate([
            df['best_fm_mase'].values[:k],
            df['best_specialist_mase'].values[k:],
        ])
        hybrid_mases.append(float(mase_vals.mean()))
        relative_costs.append(1.0 + alpha * (FM_COST_MULTIPLIER - 1))

    # Find Pareto frontier (non-dominated points: lower cost AND lower MASE)
    # — since cost is monotonically increasing with alpha, every point on the
    #   curve that achieves a lower MASE than all cheaper alternatives is Pareto.
    pareto_mask = np.zeros(len(alphas), dtype=bool)
    best_mase_so_far = float('inf')
    for i in range(len(alphas)):
        if hybrid_mases[i] < best_mase_so_far:
            pareto_mask[i]   = True
            best_mase_so_far = hybrid_mases[i]

    # Identify knee point (max improvement per unit cost)
    mase_arr = np.array(hybrid_mases)
    cost_arr = np.array(relative_costs)
    # Normalize both axes to [0,1]
    mase_n = (mase_arr - mase_arr.min()) / max(mase_arr.max() - mase_arr.min(), 1e-12)
    cost_n = (cost_arr - cost_arr.min()) / max(cost_arr.max() - cost_arr.min(), 1e-12)
    dist   = np.sqrt(mase_n ** 2 + cost_n ** 2)
    knee_i = int(np.argmin(dist))

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(relative_costs, hybrid_mases, color='steelblue',
            linewidth=1.5, label='Hybrid router (sweep α)')
    ax.scatter(np.array(relative_costs)[pareto_mask],
               np.array(hybrid_mases)[pareto_mask],
               color='steelblue', s=15, zorder=3)

    # Knee
    ax.scatter([relative_costs[knee_i]], [hybrid_mases[knee_i]],
               color='orange', s=80, zorder=5,
               label=f'Knee (α={alphas[knee_i]:.2f}, '
                     f'cost={relative_costs[knee_i]:.0f}×)')

    # Baselines
    ax.axhline(pure_specialist_mase, color='green', linestyle='--',
               linewidth=1.2,
               label=f'Pure specialist (cost=1×, MASE={pure_specialist_mase:.4f})')
    ax.axhline(pure_fm_mase, color='red', linestyle='--',
               linewidth=1.2,
               label=f'Pure FM (cost={FM_COST_MULTIPLIER}×, MASE={pure_fm_mase:.4f})')

    ax.set_xscale('log')
    ax.set_xlabel('Relative Inference Cost (log scale; 1× = all specialists)',
                  fontsize=10)
    ax.set_ylabel('Mean MASE (hybrid router)', fontsize=10)
    ax.set_title('Pareto Frontier: Hybrid MASE vs Inference Cost\n'
                 f'FM cost = {FM_COST_MULTIPLIER}× specialist; '
                 'series sorted by FM advantage (best first)',
                 fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    return alphas[knee_i], relative_costs[knee_i], hybrid_mases[knee_i]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Complexity Router Analysis ===\n", flush=True)

    os.makedirs(_FIG_DIR, exist_ok=True)

    print("Loading data...", flush=True)
    features, per_series = load_data()

    print("Building routing table...", flush=True)
    routing_df = build_routing_table(features, per_series)

    if routing_df.empty:
        sys.exit("ERROR: Routing table is empty after merge. "
                 "Check that series_ids match between features and per_series_results.")

    # --- Plot 1: feature analysis ---
    feat_pdf_path = os.path.join(_FIG_DIR, 'routing_feature_analysis.pdf')
    print(f"\nWriting feature analysis → {feat_pdf_path}", flush=True)
    with pdf_backend.PdfPages(feat_pdf_path) as pdf:
        thresholds = plot_feature_analysis(routing_df.copy(), pdf)

    # --- Threshold report ---
    print("\n--- FM Win Rate ≥ 60% Thresholds ---")
    for feat in FEATURE_COLS:
        t = thresholds.get(feat)
        if t is not None:
            print(f"  {FEATURE_LABELS[feat]:<35} threshold ≥ {t:.4f}")
        else:
            print(f"  {FEATURE_LABELS[feat]:<35} win rate never reaches 60%")

    # --- Plot 2: Pareto frontier ---
    pareto_pdf_path = os.path.join(_FIG_DIR, 'pareto_frontier.pdf')
    print(f"\nWriting Pareto frontier → {pareto_pdf_path}", flush=True)
    with pdf_backend.PdfPages(pareto_pdf_path) as pdf:
        knee_alpha, knee_cost, knee_mase = plot_pareto_frontier(routing_df, pdf)

    print(f"\n--- Pareto Knee ---")
    print(f"  Optimal alpha  : {knee_alpha:.2f}  "
          f"(route top {int(knee_alpha*100)}% of series by FM advantage to FM)")
    print(f"  Relative cost  : {knee_cost:.1f}×")
    print(f"  Hybrid MASE    : {knee_mase:.4f}")

    print(f"\nDone. Figures saved to {_FIG_DIR}/", flush=True)


if __name__ == '__main__':
    main()
