# Public-data replication notes

The scripts in this repository accept either:

1. a wide return CSV: `Date,AGG.P,DBC.P,GLD.P,IWM.P,LQD.P,MUB.P,VTI.P`, or
2. a long price CSV: `RIC,Date,Price`.

For a fully public replication, download adjusted daily closes for the seven ETFs from Yahoo Finance, Stooq, Nasdaq, or another public source, convert them to the wide-return format, and run:

```bash
export PYTHONPATH=src:.
python scripts/run_full_oos_fair_eval.py \
  --data_path data/etfs_aligned_returns_wide.csv \
  --out_dir outputs/full_oos_fair_eval \
  --start_date 2019-01-02 \
  --end_date 2024-12-31 \
  --encoder cnn_lstm \
  --fqi_iters 1 \
  --epochs_per_iter 1 \
  --hidden 16 \
  --switch_thresholds 0 0.1 0.2
```

In this runtime, outbound DNS was unavailable, so the experiment was run on the already-aligned seven-ETF return panel. The raw Wind/Boke data are not redistributed.
