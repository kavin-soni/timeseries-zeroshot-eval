# Assessing the Operational Viability of Foundation Models for Time Series Forecasting

**Kavin Soni, Debanshu Das, Vamshi Guduguntla** — Google

[arXiv Paper](https://arxiv.org/abs/2605.24381) | [Dataset Sources](#datasets)

## Overview

This repository contains the code and evaluation framework for our applied study comparing foundation models against supervised baselines for time series forecasting. We evaluate three foundation models (TimesFM 2.0, TimesFM 2.5, Chronos) against four supervised baselines (XGBoost, LSTM, PatchTST, DLinear) across four operational regimes.

## Models Evaluated

| Model | Type | Parameters |
|-------|------|------------|
| TimesFM 2.0 | Foundation (zero-shot) | 500M |
| TimesFM 2.5 | Foundation (zero-shot) | 200M |
| Chronos T5-Base | Foundation (zero-shot) | 200M |
| XGBoost | Supervised | — |
| LSTM | Supervised | — |
| PatchTST | Supervised | — |
| DLinear | Supervised | — |

## Datasets

| Dataset | Domain | Freq | Series |
|---------|--------|------|--------|
| Traffic (PeMS) | Transportation | Hourly | 862 |
| ETTh1 | Energy | Hourly | 7 |
| Exchange Rate | Finance | Daily | 8 |
| M4 (Daily) | General | Daily | 4227 |

### Dataset Download Links

**Traffic (Long-Horizon PeMS)** — 862 freeway sensor lanes, San Francisco Bay Area
- Official source: https://pems.dot.ca.gov/
- Commonly used preprocessed version: https://github.com/thuml/Time-Series-Library

**ETTh1 (Electricity Transformer Temperature)**
- Official source (Informer authors): https://github.com/zhouhaoyi/ETDataset

**Exchange Rate (LSTNet)**
- Official source (Lai et al. 2018): https://github.com/laiguokun/multivariate-time-series-data (`exchange_rate` folder)

**M4 Daily**
- Official M4 competition: https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
- Kaggle mirror: https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset

## Installation

### Supervised baselines (CPU, runs locally)

```bash
pip install xgboost pandas numpy scipy scikit-learn matplotlib tensorflow torch
```

### Foundation models (GPU required, Colab recommended)

```bash
# TimesFM 2.5 and Chronos
git clone https://github.com/google-research/timesfm.git
cd timesfm && pip install -e . && cd ..
pip install chronos-forecasting

# TimesFM 2.0 requires Python 3.10 (legacy package)
# pip install timesfm  # in a Python 3.10 environment only
```

## Data Preparation

Download the raw dataset files and place them in the repo root, then run:

```bash
python data_preprocessing.py
```

This converts each dataset from wide to long format and generates the train/test CSV splits used by all model scripts.

| Dataset | Raw File(s) |
|---------|-------------|
| Traffic (PeMS) | traffic.csv |
| ETTh1 | ETTh1.csv |
| Exchange Rate | exchange_rate.csv |
| M4 Daily | Daily-train.csv, Daily-test.csv, M4-info.csv |

## Running Experiments

### Supervised baselines (run locally)

```bash
python run_xgboost.py
python run_lstm.py
python run_patchtst_all.py
python run_dlinear_all.py
```

### Foundation models (run on Colab with GPU)

```bash
python run_foundation_models.py --model timesfm25 --dataset traffic
python run_foundation_models.py --model timesfm25 --dataset m4
python run_foundation_models.py --model chronos --dataset traffic
python run_foundation_models.py --model chronos --dataset m4

# TimesFM 2.0 requires a separate Python 3.10 environment
# See run_foundation_models.py docstring for instructions
```

### Complexity Router analysis

```bash
python results/extract_series_features.py
python results/routing_analysis.py          # full-dataset analysis
python results/routing_analysis_holdout.py  # calibration/holdout split (paper results)
# Figures saved to figures/
```

## Complexity Router Evaluation

The Complexity Router determines whether to send a series to a foundation model (FM) or a supervised specialist based on four time-series features. Two scripts implement this analysis:

**`results/routing_analysis.py`** — Full-dataset analysis across all 5,089 series (862 Traffic + 4,227 M4). Derives routing feature thresholds and generates the FM win-rate decile figure (`figures/routing_feature_analysis.pdf`). Used for exploratory analysis only; thresholds and Pareto results are in-sample.

**`results/routing_analysis_holdout.py`** — Calibration/holdout split for out-of-sample evaluation. All paper-reported numbers (MASE 0.964, α=0.29, 70% cost reduction) come from this script.

| Split | Series | Purpose |
|-------|--------|---------|
| Calibration (40%) | 2,036 (345 Traffic + 1,691 M4) | Threshold derivation |
| Holdout (60%) | 3,053 (517 Traffic + 2,536 M4) | Pareto frontier evaluation |

Random seed: 42 (set via `np.random.default_rng(42)` for reproducibility).

**Calibration routing thresholds** (FM wins when):

| Feature | Condition |
|---------|-----------|
| Spectral Entropy | ≥ 0.1471 |
| Coefficient of Variation | ≥ 0.2132 |
| Seasonal Autocorrelation | < 0.9048 or ≥ 0.9902 |
| Trend Strength (R²) | < 0.0136 |

**Holdout Pareto knee**: α=0.29, cost=296×, MASE=0.964, 70.4% cost reduction vs pure FM deployment.

## Results

See Table 2 in the paper for full results. Key findings:

- Foundation models outperform all supervised baselines on Traffic (periodic, high-frequency)
- XGBoost dominates on Energy (physically constrained)
- TimesFM 2.5 leads on Exchange Rate, suggesting newer FM architectures are closing the gap in stochastic domains
- The Complexity Router at α=0.29 achieves MASE=0.964 on the holdout set at 296× inference cost, a 70% reduction vs pure FM deployment (evaluated via `routing_analysis_holdout.py`)

## Repository Structure

```
├── run_foundation_models.py        # TimesFM 2.0, 2.5, Chronos
├── run_xgboost.py                  # XGBoost baseline
├── run_lstm.py                     # LSTM baseline
├── run_patchtst_all.py             # PatchTST baseline
├── run_dlinear_all.py              # DLinear baseline
├── results/
│   ├── extract_series_features.py      # Feature extraction
│   ├── routing_analysis.py             # Full-dataset router analysis
│   ├── routing_analysis_holdout.py     # Calibration/holdout split (paper results)
│   └── per_series_results.csv          # Per-series MASE
├── figures/                        # Generated plots
└── data/                           # Dataset CSVs
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{soni2026foundation,
  title={Assessing the Operational Viability of Foundation Models for Time Series Forecasting},
  author={Soni, Kavin and Das, Debanshu and Guduguntla, Vamshi},
  journal={arXiv preprint},
  year={2026}
}
```
