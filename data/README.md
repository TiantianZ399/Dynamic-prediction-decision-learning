# Data directory

Raw Wind/Boke market data are not included in this public repository.

Expected real-data file for experiments:

```text
data/etfs_aligned_returns_wide.csv
```

Format:

```csv
Date,AGG.P,DBC.P,GLD.P,IWM.P,LQD.P,MUB.P,VTI.P
2011-01-03,0.0012,...
```

You can also prepare data from long price format:

```bash
python scripts/prepare_wind_etf_data.py \
  --input /path/to/long_price_file.csv \
  --output data/etfs_aligned_returns_wide.csv
```
