# Public Yahoo seven-ETF replication: full OOS result

## Scope and reproducibility

This is a public-data replication of the seven-ETF full OOS diagnostic, not a rerun of the proprietary Wind/Boke panel. The input uses Yahoo Finance adjusted daily prices downloaded through `yfinance==1.5.1`, maps the seven Yahoo tickers to the repository's `.P` labels, aligns only common observed dates, and applies no price or return imputation.

- Price sample: 2008-01-02 through 2024-12-31.
- Return sample: 2008-01-03 through 2024-12-31; 4,278 daily observations.
- OOS evaluation: 2019-01-03 through 2024-12-17; 75 rolling blocks and 1,500 daily decisions.
- Model: CNN-LSTM Conservative FQI proxy; 250-day train lookback, 120-day state sequence, 20-day test block, one FQI iteration, one epoch per iteration, hidden size 16, seed 2021.
- Costs: 20 basis points times L1 turnover in both reward and evaluation.

The Colab-generated manifest is committed at `artifacts/public_yahoo_colab_f492551/etfs_original_7_20080102_20241231_manifest.json`. It records the exact experiment checkout as `f49255135ad7e782c6d600a092330474ee5ff545` on `codex/public-yahoo-replication`; `outputs/full_oos/code_version.json` records the same SHA. The executed notebook is `notebooks/01_download_public_etf_data_colab.executed.ipynb`.

The full OOS result was run in Colab from that GitHub checkout, following the completed three-roll smoke test. The source notebook deliberately exports and validates the data before printing experiment commands; `scripts/runpublic` provides the corresponding smoke/full launcher and writes the code version into both the manifest and full-output directory.

## Command

```bash
python scripts/runpublic smoke
python scripts/runpublic full
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

The committed Colab result summary files are in `artifacts/public_yahoo_colab_f492551/outputs/full_oos/`. Run the following after the OOS command to generate editable SVG/PDF, 600 dpi TIFF, and PNG preview files from the same daily curves:

```bash
python scripts/plot_full_oos_results.py \
  --results-dir outputs/public_original_7_full
```

The source curves and summary tables are in `outputs/public_original_7_full/`; the figure is written under `outputs/public_original_7_full/figures/`.
