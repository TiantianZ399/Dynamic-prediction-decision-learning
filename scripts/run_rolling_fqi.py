#!/usr/bin/env python3
from __future__ import annotations
import argparse
from dynport.fqi import FQIConfig, rolling_fqi

def main():
    p = argparse.ArgumentParser(description="Run rolling finite-action FQI proxy.")
    p.add_argument("--data_path", required=True); p.add_argument("--out_dir", default="outputs/fqi")
    p.add_argument("--start_date", default="2019-01-02"); p.add_argument("--end_date", default="2024-12-31")
    p.add_argument("--train_len", type=int, default=250); p.add_argument("--test_len", type=int, default=20)
    p.add_argument("--warmup", type=int, default=120); p.add_argument("--vol_window", type=int, default=20)
    p.add_argument("--eval_cost", type=float, default=0.002); p.add_argument("--reward_cost", type=float, default=0.002); p.add_argument("--risk_coef", type=float, default=0.25)
    p.add_argument("--fqi_iters", type=int, default=3); p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--gamma_values", type=float, nargs="+", default=[0.0, 0.95]); p.add_argument("--max_rolls", type=int, default=12)
    p.add_argument("--run_extra_trees", action="store_true"); p.add_argument("--seed", type=int, default=2020)
    a = p.parse_args()
    cfg = FQIConfig(start_date=a.start_date, end_date=a.end_date, train_len=a.train_len, test_len=a.test_len, warmup=a.warmup, vol_window=a.vol_window, eval_cost=a.eval_cost, reward_cost=a.reward_cost, risk_coef=a.risk_coef, fqi_iters=a.fqi_iters, ridge_alpha=a.ridge_alpha, gamma_values=tuple(a.gamma_values), run_extra_trees=a.run_extra_trees, max_rolls=a.max_rolls, seed=a.seed)
    res = rolling_fqi(a.data_path, a.out_dir, cfg)
    print(res[["method", "n_days", "net_nav", "net_sharpe", "gross_sharpe", "max_drawdown", "avg_turnover"]].to_string(index=False))
    print(f"Saved results to {a.out_dir}")
if __name__ == "__main__": main()
