# Public Yahoo seven-ETF replication: full OOS result

## Scope and reproducibility

This is a public-data replication of the seven-ETF full OOS diagnostic, not a rerun of the proprietary Wind/Boke panel. The input uses Yahoo Finance adjusted daily prices downloaded through `yfinance==1.5.1`, maps the seven Yahoo tickers to the repository's `.P` labels, aligns only common observed dates, and applies no price or return imputation.

- Price sample: 2008-01-02 through 2024-12-31.
- Return sample: 2008-01-03 through 2024-12-31; 4,278 daily observations.
- OOS evaluation: 2019-01-03 through 2024-12-17; 75 rolling blocks and 1,500 daily decisions.
- Model: CNN-LSTM Conservative FQI proxy; 250-day train lookback, 120-day state sequence, 20-day test block, one FQI iteration, one epoch per iteration, hidden size 16, seed 2021.
- Costs: 20 basis points times L1 turnover in both reward and evaluation.

The data manifest is `data/etfs_original_7_20080102_20241231_manifest.json` (SHA-256 `8e0d5e33084be4e5ea99d1602bd2f7971e66eb9be711b51a84589f45c228bbe9`). The complete public-data bundle is `public_etf_data_original_7_20080102_20241231.zip` (SHA-256 `f37a8102c04fc0e9511e4d391e1a473a167f3417bd6d7b41a0a9c1c46ef9bf71`).

The result was run with the complete local replication package, which includes `scripts/run_full_oos_fair_eval.py`. At the time of verification, the public GitHub `main` commit `05dad79b89c770802162144f9baff1bdc5983837` did not include that runner. Therefore, Colab users must use the complete replication package or first merge the full-OOS runner into the public repository; the notebook intentionally stops after its smoke command when the cloned checkout lacks the runner.

## Command

```bash
export PYTHONPATH=src:.
.venv/bin/python scripts/run_full_oos_fair_eval.py \
  --data_path data/etfs_aligned_returns_wide.csv \
  --out_dir outputs/full_oos_original_7 \
  --start_date 2019-01-02 --end_date 2024-12-31 \
  --encoder cnn_lstm --train_len 250 --test_len 20 --seq_len 120 \
  --fqi_iters 1 --epochs_per_iter 1 --hidden 16 \
  --eval_cost 0.002 --reward_cost 0.002 --risk_coef 0.25 \
  --switch_thresholds 0 0.1 0.2 --max_rolls -1 --seed 2021
```

## Aggregate performance

| Method | Net NAV | Net Sharpe | Max drawdown | Avg. turnover |
|---|---:|---:|---:|---:|
| EW zero-cost target | 1.6280 | 0.8727 | -0.2065 | 0.0000 |
| EW monthly rebalanced | 1.6034 | 0.8537 | -0.2069 | 0.0012 |
| EW daily rebalanced | 1.6010 | 0.8444 | -0.2072 | 0.0056 |
| EW buy-and-hold | 1.6287 | 0.8410 | -0.2070 | 0.0000 |
| CNN-LSTM CQL-FQI, κ = 0.2 | 1.5667 | 0.5985 | -0.3253 | 0.1320 |
| CNN-LSTM CQL-FQI, κ = 0.1 | 0.5856 | -0.4423 | -0.4985 | 0.2390 |
| CNN-LSTM CQL-FQI, κ = 0 | 0.4459 | -0.6828 | -0.6135 | 0.3253 |

For the best learned threshold in this run (κ = 0.2), active information ratios range from -0.0168 versus equal-weight buy-and-hold to 0.0170 versus daily-rebalanced equal weight. Its active NAV is below one against every equal-weight variant.

## Interpretation and limitations

The high threshold avoids the severe turnover and drawdowns of lower thresholds, but it does not beat equal weight on terminal NAV or risk-adjusted performance in this single-seed Yahoo replication. This result must be described as a public-data replication, not as a numerical confirmation of the proprietary-data result or as evidence of superior performance.

The outputs contain no multi-seed uncertainty interval or cost sensitivity analysis. Those are separate robustness experiments and should not be inferred from this run.

## Figure and output provenance

Run the following after the OOS command to generate editable SVG/PDF, 600 dpi TIFF, and PNG preview files from the same daily curves:

```bash
.venv/bin/python scripts/plot_full_oos_results.py \
  --results-dir outputs/full_oos_original_7
```

The source curves and summary tables are in `outputs/full_oos_original_7/`; the figure is written under `outputs/full_oos_original_7/figures/`.
