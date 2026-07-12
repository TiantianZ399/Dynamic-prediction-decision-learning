# Yahoo public-data replication

`notebooks/01_download_public_etf_data_colab.ipynb` creates the public seven-ETF input used for the first replication pass. It downloads adjusted daily prices from Yahoo Finance, aligns the seven assets on their observed-date intersection, computes simple returns, and writes the repository input at `data/etfs_aligned_returns_wide.csv`.

## Universe and data contract

The default `original_7` universe preserves the labels required by the candidate-action code:

| Repository column | Yahoo ticker |
|---|---|
| `AGG.P` | `AGG` |
| `DBC.P` | `DBC` |
| `GLD.P` | `GLD` |
| `IWM.P` | `IWM` |
| `LQD.P` | `LQD` |
| `MUB.P` | `MUB` |
| `VTI.P` | `VTI` |

The notebook uses `auto_adjust=True`, an inclusive sample through 2024-12-31, intersection alignment, and no price or return imputation. Retain the dated snapshot, manifest, hashes, ticker mapping, and quality files with every experiment output.

## Run order

1. Run the notebook through the repository-loader verification cell.
2. Run the printed three-roll command. This is an environment and interface smoke test only.
3. If the cloned commit contains `scripts/run_full_oos_fair_eval.py`, run the printed full command. Its output goes to `outputs/full_oos_original_7`.
4. If that script is absent, stop after the smoke test and obtain the commit containing the full-OOS runner. Do not report the smoke-test result as a public replication.

The notebook prints the cloned checkout's commit SHA and the commands for the detected runner. The formal public replication must preserve that SHA, the notebook's generated manifest, and the console log.

## Expected full-OOS artifacts

The formal run should create the four equal-weight baseline curves, three CNN-LSTM CQL threshold curves, `aggregate_results_full_oos.csv`, `active_metrics_vs_equal_weight.csv`, `action_counts.csv`, figures, and `metadata.json`. Check the actual covered dates, rolling-block count, and OOS decision count rather than assuming they exactly match the proprietary-data run.

Yahoo data can differ from the original proprietary Wind/Boke panel because of vendor adjustments and date alignment. Describe results as a public-data replication and compare them structurally; do not claim numerical equality or superiority over equal weight solely from terminal NAV.
