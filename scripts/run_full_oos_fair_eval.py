#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch

from dynport.data import load_returns
from dynport.actions import ACTION_NAMES, ActionConfig, make_candidate_actions, action_weight
from dynport.metrics import performance_metrics, add_nav
from dynport.deep_temporal_cql_fqi import (
    DeepTemporalCQLConfig,
    _standardize_returns,
    _hist,
    build_transition_data,
    train_model,
)


def _date_str(x):
    return pd.Timestamp(x).date().isoformat()


def _perf_from_df(df, net_col='net_return', gross_col='gross_return', turnover_col='turnover'):
    return performance_metrics(df[net_col], df[gross_col] if gross_col in df else None, df[turnover_col] if turnover_col in df else None)


def equal_weight_baselines(dates, returns, start, end, cost=0.002):
    """Compute fair equal-weight variants on the exact OOS day set.

    The loop follows the same date convention as the RL evaluator: at index t we
    choose weights before seeing returns[t+1], and the row date is dates[t+1].
    """
    N = returns.shape[1]
    eq = np.ones(N) / N
    rows = {"ew_zero_cost_target": [], "ew_buy_hold": [], "ew_daily_rebalanced": [], "ew_monthly_rebalanced": []}

    # Zero-cost target EW: the old diagnostic baseline, kept for transparency.
    for t in range(start, min(end, len(returns) - 1)):
        r = returns[t + 1]
        gross = float(eq @ r)
        rows["ew_zero_cost_target"].append({
            "Date": _date_str(dates[t + 1]), "method": "ew_zero_cost_target",
            "gross_return": gross, "net_return": gross, "turnover": 0.0, "action": "equal_weight"
        })

    # Buy-and-hold from equal initial weights, no turnover after the initial position.
    w_bh = eq.copy()
    for t in range(start, min(end, len(returns) - 1)):
        r = returns[t + 1]
        gross = float(w_bh @ r)
        rows["ew_buy_hold"].append({
            "Date": _date_str(dates[t + 1]), "method": "ew_buy_hold",
            "gross_return": gross, "net_return": gross, "turnover": 0.0, "action": "buy_hold"
        })
        denom = 1.0 + gross
        if denom > 1e-12:
            w_bh = w_bh * (1.0 + r) / denom
            w_bh = np.clip(w_bh, 0, None); w_bh = w_bh / w_bh.sum()

    # Daily rebalanced EW: rebalance at the start of each day to equal weight;
    # after daily returns, weights drift and create next-day turnover.
    w_after = eq.copy()
    for t in range(start, min(end, len(returns) - 1)):
        r = returns[t + 1]
        turnover = float(np.abs(eq - w_after).sum())
        w = eq.copy()
        gross = float(w @ r)
        rows["ew_daily_rebalanced"].append({
            "Date": _date_str(dates[t + 1]), "method": "ew_daily_rebalanced",
            "gross_return": gross, "net_return": gross - cost * turnover,
            "turnover": turnover, "action": "daily_rebalance"
        })
        denom = 1.0 + gross
        if denom > 1e-12:
            w_after = w * (1.0 + r) / denom
            w_after = np.clip(w_after, 0, None); w_after = w_after / w_after.sum()

    # Monthly rebalanced EW: rebalance to equal at first day and month changes.
    w_after = eq.copy()
    prev_month = None
    for t in range(start, min(end, len(returns) - 1)):
        cur_date = pd.Timestamp(dates[t + 1])
        r = returns[t + 1]
        rebalance = (prev_month is None) or (cur_date.to_period('M') != prev_month)
        if rebalance:
            w = eq.copy()
            turnover = float(np.abs(w - w_after).sum()) if prev_month is not None else 0.0
        else:
            w = w_after.copy()
            turnover = 0.0
        gross = float(w @ r)
        rows["ew_monthly_rebalanced"].append({
            "Date": _date_str(dates[t + 1]), "method": "ew_monthly_rebalanced",
            "gross_return": gross, "net_return": gross - cost * turnover,
            "turnover": turnover, "action": "monthly_rebalance" if rebalance else "hold"
        })
        denom = 1.0 + gross
        if denom > 1e-12:
            w_after = w * (1.0 + r) / denom
            w_after = np.clip(w_after, 0, None); w_after = w_after / w_after.sum()
        prev_month = cur_date.to_period('M')

    return {k: add_nav(pd.DataFrame(v)) for k, v in rows.items()}


def eval_model_continuous(model, method, dates, returns, standardized_returns, base_actions, start, end, cfg, threshold, prev):
    N = returns.shape[1]
    rows = []
    counts = {a: 0 for a in ACTION_NAMES}
    model.eval()
    with torch.no_grad():
        for t in range(start, min(end, len(returns) - 1)):
            h = torch.from_numpy(_hist(standardized_returns, t, cfg).reshape(1, cfg.seq_len, N))
            p = torch.from_numpy(prev.reshape(1, N).astype(np.float32))
            q, pred = model(h, p)
            qv = q.squeeze(0).cpu().numpy()
            best = int(np.argmax(qv))
            action = best if qv[best] - qv[0] > threshold else 0
            w = action_weight(action, t, prev, base_actions)
            r = returns[t + 1]
            gross = float(w @ r)
            turnover = float(np.abs(w - prev).sum())
            rows.append({
                "Date": _date_str(dates[t + 1]),
                "method": method,
                "gross_return": gross,
                "net_return": gross - cfg.eval_cost * turnover,
                "turnover": turnover,
                "action": ACTION_NAMES[action],
                "best_action": ACTION_NAMES[best],
                "q_hold": float(qv[0]),
                "q_best": float(qv[best]),
            })
            counts[ACTION_NAMES[action]] += 1
            prev = w.copy()
    return add_nav(pd.DataFrame(rows)), counts, prev


def active_metrics(strategy_df, baseline_df, baseline_name):
    a = strategy_df[["Date", "net_return"]].rename(columns={"net_return": "strategy_net_return"})
    b = baseline_df[["Date", "net_return"]].rename(columns={"net_return": "baseline_net_return"})
    m = a.merge(b, on="Date", how="inner")
    m["active_return"] = m["strategy_net_return"] - m["baseline_net_return"]
    x = m["active_return"].to_numpy(float)
    nav = np.cumprod(1 + x)
    ir = float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(252)) if len(x) > 1 else np.nan
    dd = (nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)
    return {
        "baseline": baseline_name,
        "n_days": int(len(x)),
        "active_nav": float(nav[-1]) if len(x) else np.nan,
        "active_mean_daily": float(x.mean()) if len(x) else np.nan,
        "active_ir": ir,
        "active_hit_rate": float((x > 0).mean()) if len(x) else np.nan,
        "active_max_drawdown": float(dd.min()) if len(x) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder", default="cnn_lstm", choices=["cnn_lstm", "tcn", "mlp"])
    ap.add_argument("--start_date", default="2019-01-02")
    ap.add_argument("--end_date", default="2024-12-31")
    ap.add_argument("--train_len", type=int, default=250)
    ap.add_argument("--test_len", type=int, default=20)
    ap.add_argument("--seq_len", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--vol_window", type=int, default=20)
    ap.add_argument("--eval_cost", type=float, default=0.002)
    ap.add_argument("--reward_cost", type=float, default=0.002)
    ap.add_argument("--risk_coef", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--fqi_iters", type=int, default=1)
    ap.add_argument("--epochs_per_iter", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--q_scale", type=float, default=100.0)
    ap.add_argument("--pred_loss_weight", type=float, default=0.05)
    ap.add_argument("--cql_alpha", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--switch_thresholds", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    ap.add_argument("--max_rolls", type=int, default=-1, help="-1 for full period")
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--torch_num_threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.torch_num_threads)))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DeepTemporalCQLConfig(
        start_date=args.start_date, end_date=args.end_date, train_len=args.train_len, test_len=args.test_len,
        warmup=args.warmup, seq_len=args.seq_len, vol_window=args.vol_window, eval_cost=args.eval_cost,
        reward_cost=args.reward_cost, risk_coef=args.risk_coef, gamma=args.gamma, fqi_iters=args.fqi_iters,
        epochs_per_iter=args.epochs_per_iter, batch_size=args.batch_size, hidden=args.hidden, lr=args.lr,
        weight_decay=args.weight_decay, q_scale=args.q_scale, pred_loss_weight=args.pred_loss_weight,
        cql_alpha=args.cql_alpha, dropout=args.dropout, encoder=args.encoder, kernel_size=args.kernel_size,
        lstm_layers=args.lstm_layers, max_rolls=None if args.max_rolls < 0 else args.max_rolls,
        switch_thresholds=tuple(args.switch_thresholds), seed=args.seed, torch_num_threads=args.torch_num_threads,
    )
    dates, assets, returns = load_returns(args.data_path, start_date="2011-01-01", end_date=args.end_date)
    ds = pd.Series(pd.to_datetime(dates))
    base_actions, covariances = make_candidate_actions(returns, assets, ActionConfig(cfg.warmup, cfg.vol_window))
    start_idx = int(np.where(ds >= pd.Timestamp(args.start_date))[0][0])
    end_idx = int(np.where(ds <= pd.Timestamp(args.end_date))[0][-1]) - 1
    roll_starts = list(range(start_idx, end_idx - cfg.test_len + 1, cfg.test_len))
    if cfg.max_rolls is not None:
        roll_starts = roll_starts[: cfg.max_rolls]
    oos_end = min(roll_starts[-1] + cfg.test_len, len(returns) - 1) if roll_starts else start_idx

    # Baseline curves over the exact covered OOS span.
    baseline_curves = equal_weight_baselines(dates, returns, start_idx, oos_end, cost=cfg.eval_cost)
    for name, df in baseline_curves.items():
        df.to_csv(out_dir / f"{name}_daily_curve.csv", index=False)

    methods = [f"{cfg.encoder}_cql_thr{thr:g}" for thr in cfg.switch_thresholds]
    method_segments = {m: [] for m in methods}
    method_prev = {m: np.ones(returns.shape[1]) / returns.shape[1] for m in methods}
    action_counts = {m: {a: 0 for a in ACTION_NAMES} for m in methods}
    roll_rows = []
    t0 = time.time()
    for roll, start in enumerate(roll_starts):
        train_idx = np.arange(start - cfg.train_len - 1, start - 1)
        train_idx = train_idx[(train_idx >= cfg.warmup) & (train_idx < len(returns) - 1)]
        if len(train_idx) == 0:
            continue
        Rs = _standardize_returns(returns, train_idx)
        train_data = build_transition_data(returns, Rs, base_actions, covariances, train_idx, cfg)
        model = train_model(train_data, cfg, returns.shape[1])
        te0, te1 = start, min(start + cfg.test_len, len(returns) - 1)
        for thr in cfg.switch_thresholds:
            m = f"{cfg.encoder}_cql_thr{thr:g}"
            df, counts, prev = eval_model_continuous(model, m, dates, returns, Rs, base_actions, te0, te1, cfg, thr, method_prev[m])
            method_prev[m] = prev
            method_segments[m].append(df)
            for k, v in counts.items():
                action_counts[m][k] += v
            met = _perf_from_df(df); met.update({"roll": roll, "method": m})
            roll_rows.append(met)
        if (roll + 1) % 10 == 0 or roll == len(roll_starts) - 1:
            print(f"Completed roll {roll+1}/{len(roll_starts)} elapsed={time.time()-t0:.1f}s", flush=True)

    all_curves = dict(baseline_curves)
    for m, segs in method_segments.items():
        if segs:
            all_curves[m] = add_nav(pd.concat(segs, ignore_index=True))
            all_curves[m].to_csv(out_dir / f"{m}_daily_curve.csv", index=False)

    # Aggregate metrics.
    agg = []
    for name, df in all_curves.items():
        met = _perf_from_df(df); met.update({"method": name})
        agg.append(met)
    agg_df = pd.DataFrame(agg).sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    agg_df.to_csv(out_dir / "aggregate_results_full_oos.csv", index=False)

    # Active metrics versus every EW variant for every non-EW method.
    active_rows = []
    for m in methods:
        if m not in all_curves: continue
        for bname, bdf in baseline_curves.items():
            row = active_metrics(all_curves[m], bdf, bname)
            row.update({"method": m})
            active_rows.append(row)
    active_df = pd.DataFrame(active_rows).sort_values(["baseline", "active_ir"], ascending=[True, False])
    active_df.to_csv(out_dir / "active_metrics_vs_equal_weight.csv", index=False)

    pd.DataFrame(roll_rows).to_csv(out_dir / "roll_level_results.csv", index=False)
    pd.DataFrame([{"method": m, **c} for m, c in action_counts.items()]).to_csv(out_dir / "action_counts.csv", index=False)
    meta = {"assets": assets, "config": asdict(cfg), "n_rolls": len(roll_starts), "covered_oos_start": _date_str(dates[start_idx+1]), "covered_oos_end": _date_str(dates[oos_end]), "elapsed_sec": time.time() - t0}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("\nAggregate results:")
    print(agg_df[["method", "n_days", "net_nav", "net_sharpe", "gross_sharpe", "max_drawdown", "avg_turnover"]].to_string(index=False))
    print("\nActive metrics:")
    print(active_df[["method", "baseline", "active_nav", "active_ir", "active_hit_rate", "active_max_drawdown"]].to_string(index=False))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
