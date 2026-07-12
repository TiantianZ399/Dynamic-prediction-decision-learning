#!/usr/bin/env python3
from __future__ import annotations
import argparse
from dynport.deep_temporal_cql_fqi import DeepTemporalCQLConfig, rolling_deep_temporal_cql_fqi


def main():
    p = argparse.ArgumentParser(description="Run rolling Deep Temporal CQL-FQI diagnostic with TCN/CNN-LSTM encoders.")
    p.add_argument("--data_path", required=True)
    p.add_argument("--out_dir", default="outputs/deep_temporal_cql_fqi")
    p.add_argument("--encoder", default="tcn", choices=["tcn", "cnn_lstm", "mlp"])
    p.add_argument("--start_date", default="2019-01-02")
    p.add_argument("--end_date", default="2024-12-31")
    p.add_argument("--train_len", type=int, default=250)
    p.add_argument("--test_len", type=int, default=20)
    p.add_argument("--warmup", type=int, default=120)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--vol_window", type=int, default=20)
    p.add_argument("--eval_cost", type=float, default=0.002)
    p.add_argument("--reward_cost", type=float, default=0.002)
    p.add_argument("--risk_coef", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--fqi_iters", type=int, default=2)
    p.add_argument("--epochs_per_iter", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--q_scale", type=float, default=100.0)
    p.add_argument("--pred_loss_weight", type=float, default=0.05)
    p.add_argument("--cql_alpha", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--max_rolls", type=int, default=12)
    p.add_argument("--switch_thresholds", type=float, nargs="+", default=[0.0, 0.10, 0.20])
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--torch_num_threads", type=int, default=1)
    a = p.parse_args()
    cfg = DeepTemporalCQLConfig(
        start_date=a.start_date, end_date=a.end_date, train_len=a.train_len, test_len=a.test_len,
        warmup=a.warmup, seq_len=a.seq_len, vol_window=a.vol_window, eval_cost=a.eval_cost,
        reward_cost=a.reward_cost, risk_coef=a.risk_coef, gamma=a.gamma, fqi_iters=a.fqi_iters,
        epochs_per_iter=a.epochs_per_iter, batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
        q_scale=a.q_scale, pred_loss_weight=a.pred_loss_weight, cql_alpha=a.cql_alpha, hidden=a.hidden,
        dropout=a.dropout, encoder=a.encoder, kernel_size=a.kernel_size, lstm_layers=a.lstm_layers,
        max_rolls=a.max_rolls, switch_thresholds=tuple(a.switch_thresholds), seed=a.seed,
        torch_num_threads=a.torch_num_threads,
    )
    res = rolling_deep_temporal_cql_fqi(a.data_path, a.out_dir, cfg)
    print(res[["method", "n_days", "net_nav", "net_sharpe", "gross_sharpe", "max_drawdown", "avg_turnover"]].to_string(index=False))
    print(f"Saved results to {a.out_dir}")


if __name__ == "__main__":
    main()
