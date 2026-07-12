# Dynamic Prediction--Decision Portfolio Optimization

This repository contains a preliminary research codebase for **dynamic portfolio optimization with a separated prediction layer and decision layer**. The project studies whether portfolio learning should be implemented as:

1. **two-stage learning**: deep prediction first, portfolio decision later;
2. **integrated learning**: prediction and optimizer trained end-to-end; or
3. **dynamic decision learning**: deep state representation plus reinforcement-learning / dynamic-programming decision layer.

The working hypothesis is that many portfolio optimizers are too static: they solve a one-period or receding-horizon allocation problem but do not explicitly learn how today’s portfolio position affects tomorrow’s transaction cost, risk, and rebalancing value. This repo therefore focuses on a modular framework:

```text
past market data -> deep state / prediction layer -> dynamic decision layer -> portfolio weights
```

The current recommended branch is a conservative finite-action offline-RL proxy, **Deep CQL-FQI**, which combines a deep state encoder, an auxiliary return-prediction head, fitted Q-iteration, and no-trade/conservative constraints.

## Status

This is a **preliminary research repository**, not a production trading system.

The initial seven-ETF experiments are diagnostic. The strongest passive benchmark, equal weight, still beats the learned methods in the short 12-roll test. However, the conservative deep-RL decision layer improves over plain FQI and over the tested two-stage/integrated baselines in that diagnostic setting. Results should be treated as a starting point for research, not as evidence of a deployable strategy.

## Data policy

Raw market data are **not included**. The research draft was initiated during an internship at Boke Simu and uses ETF data sourced from Wind/Boke City infrastructure. Because the raw data may be proprietary, this public repo only includes:

- code;
- synthetic sample data;
- preliminary aggregate result tables;
- paper source/draft.

To reproduce the real-data experiments, provide your own CSV file in one of the supported formats.

## Supported data formats

### Wide returns format

```csv
Date,AGG.P,DBC.P,GLD.P,IWM.P,LQD.P,MUB.P,VTI.P
2011-01-03,0.0012,-0.0031,...
```

### Long price format

A headerless or headered CSV with columns:

```text
RIC, Date, Price
```

Dates can be parsed by pandas. The loader will compute percentage returns and pivot to a Date x Asset return matrix.

## Quick start with synthetic data

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

python scripts/generate_synthetic_data.py --out examples/synthetic_7etf_returns.csv
python scripts/run_rolling_fqi.py \
  --data_path examples/synthetic_7etf_returns.csv \
  --out_dir outputs/fqi_synthetic \
  --max_rolls 3

python scripts/run_deep_cql_fqi.py \
  --data_path examples/synthetic_7etf_returns.csv \
  --out_dir outputs/deep_cql_synthetic \
  --max_rolls 3 \
  --epochs_per_iter 1 \
  --fqi_iters 1
```

## Reproducing the seven-ETF diagnostic

Place the real ETF return data at `data/etfs_aligned_returns_wide.csv`, then run:

```bash
export PYTHONPATH=src
python scripts/run_deep_cql_fqi.py \
  --data_path data/etfs_aligned_returns_wide.csv \
  --out_dir outputs/deep_cql_7etf \
  --start_date 2019-01-02 \
  --end_date 2024-12-31 \
  --train_len 250 \
  --test_len 20 \
  --seq_len 120 \
  --max_rolls 12 \
  --eval_cost 0.002 \
  --reward_cost 0.002 \
  --risk_coef 0.25
```


## Deep temporal encoder upgrade

The repository now includes a temporal deep-learning version of the conservative CQL-FQI decision layer:

```bash
python scripts/run_deep_temporal_cql_fqi.py \
  --data_path data/etfs_aligned_returns_wide.csv \
  --out_dir outputs/deep_temporal_cql_7etf \
  --encoder cnn_lstm \
  --start_date 2019-01-02 \
  --end_date 2024-12-31 \
  --train_len 250 \
  --test_len 20 \
  --seq_len 120 \
  --max_rolls 12 \
  --hidden 16 \
  --fqi_iters 1 \
  --epochs_per_iter 1 \
  --switch_thresholds 0.0 0.10 0.20
```

Supported encoders:

```text
--encoder mlp       # flattened 120-day history baseline
--encoder tcn       # causal temporal convolution network
--encoder cnn_lstm  # convolutional feature extractor followed by LSTM
```

The current recommended temporal encoder for the seven-ETF diagnostic is `cnn_lstm`. It performed best among the fast temporal encoder tests, although equal weight remains the strongest benchmark in the short 12-roll diagnostic.

## Repository layout

```text
src/dynport/            Core data, action, metric, FQI, and Deep CQL-FQI utilities
scripts/                Command-line experiment scripts
configs/                Example configuration file
docs/                   Research notes and preliminary result summary
paper/                  Preliminary paper draft and arXiv source
examples/               Synthetic data fixture
tests/                  Minimal smoke tests
```

## Preliminary result summary

The 12-roll seven-ETF diagnostic used 250-day rolling lookback windows, 20-day test blocks, past 120-day state windows, and 20 bps L1 turnover cost. The best Deep CQL-FQI variant improved over the tested two-stage and integrated baselines, but equal weight remained strongest:

| Method | Net NAV | Net Sharpe | Avg Turnover |
|---|---:|---:|---:|
| Equal weight | 1.1439 | 2.7689 | 0.0000 |
| Deep CQL-FQI, threshold 0.20 | 1.1078 | 1.6897 | 0.0213 |
| Two-stage, H=10 | 1.0740 | 1.4769 | 0.1226 |
| Integrated, H=10 | 1.0572 | 1.1455 | 0.1549 |
| Plain FQI, gamma=0 | 1.0821 | 0.6968 | 0.2970 |
| Plain FQI, gamma=0.95 | 0.9949 | 0.0299 | 0.2589 |

These results are not a final claim. They motivate the next research step: a full 2019--2024 walk-forward study, stronger passive baselines, fair rebalancing-cost accounting, and more robust offline-RL regularization.

## Research context

Recent DRL portfolio work formulates portfolio selection as a Markov decision process and embeds risk aversion and transaction costs into the reward. The TD3-based portfolio paper by Jiang, Olmo, and Atwi (2024) proposes a model-free DRL framework using an extended Markowitz mean-variance reward with transaction costs and risk aversion. This repo follows the same high-level MDP motivation but starts from offline FQI / conservative FQI because the current seven-ETF setting is small and data-limited.

## Disclaimer

This repository is for academic research only. It is not investment advice, not a trading recommendation, and not a production-ready portfolio system.

## Full OOS fair equal-weight evaluation

This release includes a full 2019--2024 OOS diagnostic with fair equal-weight variants and active-return metrics. See:

- `docs/full_oos_fair_eval_report.md`
- `outputs/full_oos_fair_eval/aggregate_results_full_oos.csv`
- `outputs/full_oos_fair_eval/active_metrics_vs_equal_weight.csv`

Run:

```bash
export PYTHONPATH=src:.
python scripts/run_full_oos_fair_eval.py \
  --data_path data/etfs_aligned_returns_wide.csv \
  --out_dir outputs/full_oos_fair_eval \
  --encoder cnn_lstm \
  --start_date 2019-01-02 \
  --end_date 2024-12-31 \
  --train_len 250 \
  --test_len 20 \
  --seq_len 120 \
  --fqi_iters 1 \
  --epochs_per_iter 1 \
  --hidden 16 \
  --switch_thresholds 0 0.1 0.2
```

### Yahoo public-data replication

Use [`notebooks/01_download_public_etf_data_colab.ipynb`](notebooks/01_download_public_etf_data_colab.ipynb) in Colab to download the seven-ETF Yahoo Finance panel and create `data/etfs_aligned_returns_wide.csv`. The notebook records the checkout commit, data provenance, and hashes, then prints a three-roll smoke command and, when available, the full 2019--2024 OOS command. See [`docs/PUBLIC_DATA_COLAB.md`](docs/PUBLIC_DATA_COLAB.md) for the required run order and reporting constraints.

The completed Yahoo public-data replication is documented in [`docs/public_yahoo_oos_report.md`](docs/public_yahoo_oos_report.md). Regenerate its figure from the evaluated daily curves with:

```bash
python scripts/plot_full_oos_results.py --results-dir outputs/full_oos_original_7
```
