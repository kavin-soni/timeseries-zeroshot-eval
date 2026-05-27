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

Download the raw dataset files from their original sources and place them in the repo root, then run:

```bash
python data_preprocessing.py
```

This converts each dataset from wide to long format and generates the train/test CSV splits used by all model scripts.

| Dataset | Source | Raw File |
|---------|--------|----------|
| Traffic (PeMS) | [LSTNet](https://github.com/laujustin/LSTNet) | traffic.csv |
| ETTh1 | [Informer](https://github.com/zhouhaoyi/ETDataset) | ETTh1.csv |
| Exchange Rate | [LSTNet](https://github.com/laujustin/LSTNet) | exchange_rate.csv |
| M4 Daily | [M4 Competition](https://github.com/Mcompetitions/M4-methods) | Daily-train.csv, Daily-test.csv, M4-info.csv |

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
python results/routing_analysis.py
# Figures saved to figures/
```

## Results

See Table 2 in the paper for full results. Key findings:

- Foundation models outperform all supervised baselines on Traffic (periodic, high-frequency)
- XGBoost dominates on Energy (physically constrained)
- TimesFM 2.5 leads on Exchange Rate, suggesting newer FM architectures are closing the gap in stochastic domains
- The Complexity Router at α=0.30 achieves better accuracy than pure FM deployment at 70% lower inference cost

## Repository Structure

```
├── run_foundation_models.py        # TimesFM 2.0, 2.5, Chronos
├── run_xgboost.py                  # XGBoost baseline
├── run_lstm.py                     # LSTM baseline
├── run_patchtst_all.py             # PatchTST baseline
├── run_dlinear_all.py              # DLinear baseline
├── results/
│   ├── extract_series_features.py  # Feature extraction
│   ├── routing_analysis.py         # Complexity Router
│   └── per_series_results.csv      # Per-series MASE
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
