# 🔮 Dynamic Prediction--Decision Portfolio Optimization

🪄 This is repo implements the prediction and decision methodology purposed in arxiv link pending and developed from internship during boke technology. 

You can see the arxiv paper in the paper section: [paper/Dynamic_Prediction_Decision_Learning.pdf](https://github.com/TiantianZ399/Dynamic-prediction-decision-learning/blob/main/paper/Dynamic_Prediction_Decision_Learning.pdf)

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
