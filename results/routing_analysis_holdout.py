"""
routing_analysis_holdout.py — Calibration/holdout split validation of the Complexity Router.

Addresses in-sample evaluation concern by deriving routing thresholds on a 40%
calibration split and evaluating the Pareto frontier on the remaining 60% holdout.

Split (np.random.seed(42)):
  Traffic : 345 calib / 517 holdout   (40/60 of 862)
  M4      : 1691 calib / 2536 holdout (40/60 of 4227)
  Total   : 2036 calib / 3053 holdout

Inputs:
  results/series_features.csv     (from extract_series_features.py)
  results/per_series_results.csv  (from all model evaluation scripts)

Outputs:
  figures/routing_feature_analysis_holdout.pdf  — decile plots on calibration set
  figures/pareto_frontier_holdout.pdf           — Pareto frontier on holdout set

Prints:
  Calibration thresholds for all four features
  Holdout Pareto knee: alpha, cost, MASE
  Holdout pure FM MASE, pure specialist MASE, cost reduction

FM models       : timesfm25, chronos
Specialist models: patchtst, dlinear
FM inference cost: 1000× a single specialist call
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

warnings.filterwarnings('ignore', category=FutureWarning)

_HERE    = os.path.dirname(os.path.abspath(__file__))
_REPO    = os.path.dirname(_HERE)
_FIG_DIR = os.path.join(_REPO, 'figures')

FEATURES_PATH   = os.path.join(_HERE, 'series_features.csv')
PER_SERIES_PATH = os.path.join(_HERE, 'per_series_results.csv')

FM_MODELS         = {'timesfm25', 'chronos'}
SPECIALIST_MODELS = {'patchtst', 'dlinear'}

FEATURE_COLS = ['spectral_entropy', 'cv', 'seasonal_autocorr', 'trend_strength']
FEATURE_LABELS = {
    'spectral_entropy':  'Spectral Entropy',
    'cv':                'Coefficient of Variation',
    'seasonal_autocorr': 'Seasonal Autocorrelation (lag S)',
    'trend_strength':    'Trend Strength (R²)',
}

# Direction each feature's FM win rate increases:
#   'increasing'  → FM wins at HIGH values → lower bound (≥ X)
#   'decreasing'  → FM wins at LOW values  → upper bound (< X)
#   'non_monotone'→ FM wins at BOTH extremes → two-sided bounds
FEATURE_DIRECTION = {
    'spectral_entropy':  'increasing',
    'cv':                'increasing',
    'seasonal_autocorr': 'non_monotone',
    'trend_strength':    'decreasing',
}

FM_COST_MULTIPLIER = 1000
N_DECILES          = 10
WIN_RATE_THRESHOLD = 0.60

CALIB_FRAC = 0.40
RANDOM_SEED = 42

# Exact counts derived from floor(frac * n)
N_TRAFFIC_CALIB = 345   # floor(0.40 * 862)
N_M4_CALIB      = 1691  # floor(0.40 * 4227)


# ---------------------------------------------------------------------------
# Data loading & routing table
# ---------------------------------------------------------------------------

def load_data():
    for path, hint in [
        (FEATURES_PATH,   "Run: python results/extract_series_features.py"),
        (PER_SERIES_PATH, "Run all model scripts with --save first."),
    ]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found.\n{hint}")

    features = pd.read_csv(FEATURES_PATH)
    features['series_id'] = features['series_id'].astype(str)

    per_series = pd.read_csv(PER_SERIES_PATH)
    per_series['series_id'] = per_series['series_id'].astype(str)
    per_series = per_series[per_series['dataset'].isin(['traffic', 'm4'])].copy()

    return features, per_series


def build_routing_table(features, per_series):
    pivot = (per_series
             .pivot_table(index=['dataset', 'series_id'],
                          columns='model',
                          values='MASE',
                          aggfunc='first'))

    fm_cols   = [c for c in pivot.columns if c in FM_MODELS]
    spec_cols = [c for c in pivot.columns if c in SPECIALIST_MODELS]

    if not fm_cols:
        sys.exit(f"ERROR: No FM models found. Found: {sorted(pivot.columns.tolist())}")
    if not spec_cols:
        sys.exit(f"ERROR: No specialist models found. Found: {sorted(pivot.columns.tolist())}")

    pivot['best_fm_mase']         = pivot[fm_cols].min(axis=1)
    pivot['best_specialist_mase'] = pivot[spec_cols].min(axis=1)
    pivot['fm_advantage']         = pivot['best_specialist_mase'] - pivot['best_fm_mase']
    pivot['fm_wins']              = (pivot['fm_advantage'] > 0).astype(int)
    pivot = pivot.reset_index()

    merged = pivot.merge(
        features[['dataset', 'series_id'] + FEATURE_COLS],
        on=['dataset', 'series_id'],
        how='inner',
    )
    print(f"  Full routing table : {len(merged)} series "
          f"({merged['fm_wins'].sum()} FM wins, "
          f"{len(merged) - merged['fm_wins'].sum()} specialist wins)")
    print(f"  FM models          : {fm_cols}")
    print(f"  Specialist models  : {spec_cols}")
    return merged


# ---------------------------------------------------------------------------
# 40/60 calibration / holdout split
# ---------------------------------------------------------------------------

def make_split(routing_df):
    rng = np.random.default_rng(RANDOM_SEED)

    traffic_ids = np.array(
        sorted(routing_df.loc[routing_df['dataset'] == 'traffic', 'series_id'].unique()))
    m4_ids = np.array(
        sorted(routing_df.loc[routing_df['dataset'] == 'm4', 'series_id'].unique()))

    traffic_calib = set(rng.choice(traffic_ids, size=N_TRAFFIC_CALIB, replace=False))
    m4_calib      = set(rng.choice(m4_ids,      size=N_M4_CALIB,      replace=False))

    def _is_calib(row):
        if row['dataset'] == 'traffic':
            return row['series_id'] in traffic_calib
        return row['series_id'] in m4_calib

    calib_mask = routing_df.apply(_is_calib, axis=1)

    calib_df   = routing_df[calib_mask].copy()
    holdout_df = routing_df[~calib_mask].copy()

    print(f"\n  Split (seed={RANDOM_SEED}):")
    for ds in ['traffic', 'm4']:
        nc = (calib_df['dataset'] == ds).sum()
        nh = (holdout_df['dataset'] == ds).sum()
        print(f"    {ds:<10}  calib={nc}  holdout={nh}")
    print(f"    {'total':<10}  calib={len(calib_df)}  holdout={len(holdout_df)}")

    return calib_df, holdout_df


# ---------------------------------------------------------------------------
# Threshold derivation — direction-aware
# ---------------------------------------------------------------------------

def _decile_table(df, feat):
    df['_decile'] = pd.qcut(df[feat], q=N_DECILES, labels=False,
                            duplicates='drop') + 1
    grouped = df.groupby('_decile').agg(
        win_rate=('fm_wins', 'mean'),
        n=('fm_wins', 'count'),
        feat_min=(feat, 'min'),
        feat_median=(feat, 'median'),
        feat_max=(feat, 'max'),
    ).reset_index().sort_values('_decile')
    df.drop(columns=['_decile'], inplace=True)
    return grouped


def derive_thresholds(calib_df):
    """
    Returns a dict keyed by feature name. Each value is a dict with:
      'direction': '>=', '<', or 'two-sided'
      For '>=' and '<': 'value' (the threshold) and 'decile' (which decile it falls in)
      For 'two-sided':  'low_upper' (FM wins when feat < X) and
                        'high_lower' (FM wins when feat >= Y)

    Rules by FEATURE_DIRECTION:
      increasing  : first decile (low→high) where win_rate >= 60% → lower bound ≥ median
      decreasing  : last consecutive decile from decile-1 where win_rate >= 75%
                    (captures the peak-signal zone); fall back to 60% if none at 75%
                    → upper bound < feat_max of that decile
      non_monotone: low-end run (consecutive from decile 1 where >= 60%) gives < feat_max,
                    high-end run (consecutive from decile 10 where >= 60%) gives >= feat_min
    """
    thresholds = {}

    for feat in FEATURE_COLS:
        g = _decile_table(calib_df.copy(), feat)
        direction = FEATURE_DIRECTION[feat]

        if direction == 'increasing':
            above = g[g['win_rate'] >= WIN_RATE_THRESHOLD]
            if not above.empty:
                r = above.iloc[0]
                thresholds[feat] = {'direction': '>=', 'value': r['feat_median'],
                                    'decile': int(r['_decile'])}
            else:
                thresholds[feat] = {'direction': '>=', 'value': None, 'decile': None}

        elif direction == 'decreasing':
            # Find last consecutive decile from decile 1 where win_rate >= 75%.
            # If none qualify at 75%, fall back to 60%.
            for cutoff in (0.75, WIN_RATE_THRESHOLD):
                last_d = None
                for _, r in g.iterrows():
                    if r['win_rate'] >= cutoff:
                        if last_d is None or r['_decile'] == last_d + 1:
                            last_d = int(r['_decile'])
                        else:
                            break
                    else:
                        break
                if last_d is not None:
                    break
            if last_d is not None:
                r = g[g['_decile'] == last_d].iloc[0]
                thresholds[feat] = {'direction': '<', 'value': r['feat_max'],
                                    'decile': last_d}
            else:
                thresholds[feat] = {'direction': '<', 'value': None, 'decile': None}

        else:  # non_monotone
            # Low-end run: consecutive from decile 1 with win_rate >= 60%
            low_last = None
            for _, r in g.iterrows():
                if r['win_rate'] >= WIN_RATE_THRESHOLD:
                    if low_last is None or r['_decile'] == low_last + 1:
                        low_last = int(r['_decile'])
                    else:
                        break
                else:
                    break

            # High-end run: consecutive from decile 10 downward with win_rate >= 60%
            high_first = None
            for _, r in g.sort_values('_decile', ascending=False).iterrows():
                if r['win_rate'] >= WIN_RATE_THRESHOLD:
                    if high_first is None or r['_decile'] == high_first - 1:
                        high_first = int(r['_decile'])
                    else:
                        break
                else:
                    break

            low_upper  = g[g['_decile'] == low_last].iloc[0]['feat_max']  if low_last  is not None else None
            high_lower = g[g['_decile'] == high_first].iloc[0]['feat_min'] if high_first is not None else None

            thresholds[feat] = {
                'direction':   'two-sided',
                'low_upper':   low_upper,   # FM wins when feat < low_upper
                'high_lower':  high_lower,  # FM wins when feat >= high_lower
                'low_decile':  low_last,
                'high_decile': high_first,
            }

    return thresholds


# ---------------------------------------------------------------------------
# Plot 1: FM win-rate by feature decile — calibration set
# ---------------------------------------------------------------------------

def plot_feature_analysis(calib_df, thresholds, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, feat in zip(axes, FEATURE_COLS):
        col_data = calib_df[feat].dropna()
        if col_data.empty:
            ax.set_title(f"{FEATURE_LABELS[feat]}\n(no data)")
            continue

        g = _decile_table(calib_df.copy(), feat)

        ax.bar(g['_decile'], g['win_rate'] * 100,
               color='steelblue', alpha=0.75, edgecolor='white')
        ax.axhline(WIN_RATE_THRESHOLD * 100, color='red', linestyle='--',
                   linewidth=1.2, label=f'{int(WIN_RATE_THRESHOLD*100)}% threshold')
        ax.set_xlabel(f"Decile of {FEATURE_LABELS[feat]}", fontsize=9)
        ax.set_ylabel("FM Win Rate (%)", fontsize=9)
        ax.set_title(FEATURE_LABELS[feat], fontsize=10, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xticks(g['_decile'])

        for _, r in g.iterrows():
            ax.text(r['_decile'], r['win_rate'] * 100 + 1.5,
                    f"n={int(r['n'])}", ha='center', va='bottom', fontsize=7)

        t = thresholds.get(feat, {})
        direction = t.get('direction')
        if direction in ('>=', '<') and t.get('decile') is not None:
            d = t['decile']
            label = (f"≥60% @ decile {d} (FM wins {direction} {t['value']:.4f})"
                     if direction == '>='
                     else f"peak @ decile {d} (FM wins {direction} {t['value']:.4f})")
            ax.axvline(d, color='orange', linestyle=':', linewidth=1.5, label=label)
        elif direction == 'two-sided':
            if t.get('low_decile') is not None:
                ax.axvline(t['low_decile'], color='orange', linestyle=':',
                           linewidth=1.5,
                           label=f"low-end: FM wins < {t['low_upper']:.4f}")
            if t.get('high_decile') is not None:
                ax.axvline(t['high_decile'], color='purple', linestyle=':',
                           linewidth=1.5,
                           label=f"high-end: FM wins ≥ {t['high_lower']:.4f}")

        ax.legend(fontsize=7)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Pareto frontier — holdout set
# ---------------------------------------------------------------------------

def plot_pareto_frontier(holdout_df, pdf):
    df = holdout_df.sort_values('fm_advantage', ascending=False).reset_index(drop=True)
    n  = len(df)

    alphas         = np.linspace(0, 1, 201)
    hybrid_mases   = []
    relative_costs = []

    pure_specialist_mase = df['best_specialist_mase'].mean()
    pure_fm_mase         = df['best_fm_mase'].mean()

    for alpha in alphas:
        k = int(round(alpha * n))
        mase_vals = np.concatenate([
            df['best_fm_mase'].values[:k],
            df['best_specialist_mase'].values[k:],
        ])
        hybrid_mases.append(float(mase_vals.mean()))
        relative_costs.append(1.0 + alpha * (FM_COST_MULTIPLIER - 1))

    pareto_mask = np.zeros(len(alphas), dtype=bool)
    best_mase_so_far = float('inf')
    for i in range(len(alphas)):
        if hybrid_mases[i] < best_mase_so_far:
            pareto_mask[i]   = True
            best_mase_so_far = hybrid_mases[i]

    mase_arr = np.array(hybrid_mases)
    cost_arr = np.array(relative_costs)
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

    mase_random = [(1 - a) * pure_specialist_mase + a * pure_fm_mase for a in alphas]
    ax.plot(relative_costs, mase_random,
            color='gray', linestyle='--', linewidth=1.2,
            label='Random router (sweep α)')

    ax.scatter([relative_costs[knee_i]], [hybrid_mases[knee_i]],
               color='orange', s=80, zorder=5,
               label=f'Knee (α={alphas[knee_i]:.2f}, '
                     f'cost={relative_costs[knee_i]:.0f}×)')

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
    ax.legend(fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    return alphas[knee_i], relative_costs[knee_i], hybrid_mases[knee_i], \
           pure_specialist_mase, pure_fm_mase


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Complexity Router — Calibration/Holdout Validation ===\n", flush=True)

    os.makedirs(_FIG_DIR, exist_ok=True)

    print("Loading data...", flush=True)
    features, per_series = load_data()

    print("Building full routing table...", flush=True)
    routing_df = build_routing_table(features, per_series)

    if routing_df.empty:
        sys.exit("ERROR: Routing table is empty after merge.")

    print("\nSplitting into calibration / holdout...", flush=True)
    calib_df, holdout_df = make_split(routing_df)

    # --- Derive thresholds from calibration set ---
    print("\nDeriving routing thresholds from calibration set...", flush=True)
    thresholds = derive_thresholds(calib_df.copy())

    # --- Plot 1: feature analysis on calibration ---
    feat_pdf_path = os.path.join(_FIG_DIR, 'routing_feature_analysis_holdout.pdf')
    print(f"\nWriting calibration feature analysis → {feat_pdf_path}", flush=True)
    with pdf_backend.PdfPages(feat_pdf_path) as pdf:
        plot_feature_analysis(calib_df.copy(), thresholds, pdf)

    print("\n--- Calibration Routing Thresholds ---")
    for feat in FEATURE_COLS:
        t = thresholds.get(feat, {})
        direction = t.get('direction')
        label = FEATURE_LABELS[feat]
        if direction == '>=':
            v = t.get('value')
            if v is not None:
                print(f"  {label:<35} FM wins when ≥ {v:.4f}  (decile {t['decile']}+)")
            else:
                print(f"  {label:<35} win rate never reaches 60%")
        elif direction == '<':
            v = t.get('value')
            if v is not None:
                print(f"  {label:<35} FM wins when <  {v:.4f}  (deciles 1–{t['decile']}, "
                      f"win rate ≥75% threshold)")
            else:
                print(f"  {label:<35} no clear upper bound found")
        elif direction == 'two-sided':
            lu = t.get('low_upper')
            hl = t.get('high_lower')
            parts = []
            if lu  is not None: parts.append(f"< {lu:.4f}  (deciles 1–{t['low_decile']})")
            if hl  is not None: parts.append(f"≥ {hl:.4f}  (deciles {t['high_decile']}–10)")
            print(f"  {label:<35} FM wins when " + " OR ".join(parts))
        else:
            print(f"  {label:<35} (no threshold derived)")

    # --- Plot 2: Pareto frontier on holdout ---
    pareto_pdf_path = os.path.join(_FIG_DIR, 'pareto_frontier_holdout.pdf')
    print(f"\nWriting holdout Pareto frontier → {pareto_pdf_path}", flush=True)
    with pdf_backend.PdfPages(pareto_pdf_path) as pdf:
        knee_alpha, knee_cost, knee_mase, pure_spec, pure_fm = \
            plot_pareto_frontier(holdout_df, pdf)

    cost_reduction_pct = (1 - knee_cost / FM_COST_MULTIPLIER) * 100

    print("\n--- Holdout Pareto Results ---")
    print(f"  Pure specialist MASE : {pure_spec:.4f}")
    print(f"  Pure FM MASE         : {pure_fm:.4f}")
    print(f"  Knee alpha           : {knee_alpha:.2f}  "
          f"(route top {int(knee_alpha * 100)}% of series to FM)")
    print(f"  Knee cost            : {knee_cost:.1f}×")
    print(f"  Knee hybrid MASE     : {knee_mase:.4f}")
    print(f"  Cost reduction vs pure FM : {cost_reduction_pct:.1f}%  "
          f"({knee_cost:.0f}× vs {FM_COST_MULTIPLIER}×)")

    print(f"\nDone. Figures saved to {_FIG_DIR}/", flush=True)


if __name__ == '__main__':
    main()
