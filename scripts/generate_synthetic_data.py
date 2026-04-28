#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
ASSETS = ["AGG.P", "DBC.P", "GLD.P", "IWM.P", "LQD.P", "MUB.P", "VTI.P"]

def generate_synthetic_returns(n_days=1800, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2011-01-03", periods=n_days)
    n = len(ASSETS); regimes = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regimes[t] = regimes[t - 1]
        if rng.random() < 0.02: regimes[t] = 1 - regimes[t]
    means = np.array([[0.00015, 0.00005, 0.00010, 0.00045, 0.00018, 0.00014, 0.00040], [0.00035, -0.00020, 0.00035, -0.00015, 0.00025, 0.00020, -0.00010]])
    vols = np.array([0.004, 0.012, 0.010, 0.016, 0.006, 0.004, 0.012])
    corr = np.full((n, n), 0.15); np.fill_diagonal(corr, 1.0)
    for i in [0, 4, 5]:
        for j in [3, 6]: corr[i, j] = corr[j, i] = -0.10
    chol = np.linalg.cholesky(np.outer(vols, vols) * corr + np.eye(n) * 1e-10)
    out = np.zeros((n_days, n))
    for t in range(n_days): out[t] = means[regimes[t]] + chol @ rng.normal(size=n)
    return dates, out

def main():
    p = argparse.ArgumentParser(description="Generate synthetic 7-ETF-like returns for smoke tests.")
    p.add_argument("--out", default="examples/synthetic_7etf_returns.csv")
    p.add_argument("--n_days", type=int, default=1800)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    dates, returns = generate_synthetic_returns(args.n_days, args.seed)
    df = pd.DataFrame(returns, columns=ASSETS); df.insert(0, "Date", dates)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {df.shape}")
if __name__ == "__main__": main()
