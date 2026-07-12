#!/usr/bin/env python3
"""Create publication-ready figures from a completed full-OOS evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

BASELINES = [
    "ew_zero_cost_target",
    "ew_monthly_rebalanced",
    "ew_daily_rebalanced",
    "ew_buy_hold",
]
POLICIES = ["cnn_lstm_cql_thr0", "cnn_lstm_cql_thr0.1", "cnn_lstm_cql_thr0.2"]
LABELS = {
    "ew_zero_cost_target": "EW target (zero cost)",
    "ew_monthly_rebalanced": "EW monthly rebalance",
    "ew_daily_rebalanced": "EW daily rebalance",
    "ew_buy_hold": "EW buy-and-hold",
    "cnn_lstm_cql_thr0": "CQL, κ = 0",
    "cnn_lstm_cql_thr0.1": "CQL, κ = 0.1",
    "cnn_lstm_cql_thr0.2": "CQL, κ = 0.2",
}
BASELINE_COLORS = ["#6B7280", "#9CA3AF", "#C4C8CF", "#374151"]
POLICY_COLORS = {"cnn_lstm_cql_thr0": "#D97706", "cnn_lstm_cql_thr0.1": "#C2410C", "cnn_lstm_cql_thr0.2": "#0F766E"}


def read_curve(results_dir: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(results_dir / f"{method}_daily_curve.csv", parse_dates=["Date"])
    required = {"Date", "net_return", "nav"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{method} curve is missing columns: {sorted(missing)}")
    return frame.sort_values("Date").reset_index(drop=True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")


def make_figure(results_dir: Path, output_dir: Path) -> Path:
    aggregate = pd.read_csv(results_dir / "aggregate_results_full_oos.csv").set_index("method")
    curves = {method: read_curve(results_dir, method) for method in BASELINES + POLICIES}
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 3.0), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.15])
    nav_ax = fig.add_subplot(grid[0, 0])
    scatter_ax = fig.add_subplot(grid[0, 1])
    active_ax = fig.add_subplot(grid[0, 2])

    for method, color in zip(BASELINES, BASELINE_COLORS):
        nav_ax.plot(curves[method]["Date"], curves[method]["nav"], color=color, lw=1.0, label=LABELS[method])
    nav_ax.plot(curves["cnn_lstm_cql_thr0.2"]["Date"], curves["cnn_lstm_cql_thr0.2"]["nav"], color=POLICY_COLORS["cnn_lstm_cql_thr0.2"], lw=1.5, label=LABELS["cnn_lstm_cql_thr0.2"])
    nav_ax.set_title("A  Full OOS net asset value", loc="left", fontweight="bold")
    nav_ax.set_ylabel("Net NAV")
    nav_ax.set_xlabel("Date")
    nav_ax.legend(fontsize=5.2, ncol=1, loc="upper left")

    for method in BASELINES + POLICIES:
        row = aggregate.loc[method]
        color = POLICY_COLORS.get(method, "#6B7280")
        marker = "o" if method in BASELINES else "D"
        size = 28 if method == "cnn_lstm_cql_thr0.2" else 18
        scatter_ax.scatter(row["avg_turnover"], row["net_sharpe"], color=color, marker=marker, s=size, zorder=3)
        if method in POLICIES:
            scatter_ax.annotate(LABELS[method].replace("CNN-LSTM ", ""), (row["avg_turnover"], row["net_sharpe"]), xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=4.7)
    scatter_ax.annotate("EW variants\n(turnover ≤ 0.006)", (0.014, 0.91), ha="left", va="top", fontsize=5.1)
    scatter_ax.axhline(0, color="#9CA3AF", lw=0.7, zorder=1)
    scatter_ax.set_title("B  Cost-aware performance", loc="left", fontweight="bold")
    scatter_ax.set_xlabel("Average daily L1 turnover")
    scatter_ax.set_ylabel("Net annualized Sharpe")

    strategy = curves["cnn_lstm_cql_thr0.2"][["Date", "net_return"]].rename(columns={"net_return": "strategy"})
    for method, color in zip(BASELINES, BASELINE_COLORS):
        baseline = curves[method][["Date", "net_return"]].rename(columns={"net_return": "baseline"})
        merged = strategy.merge(baseline, on="Date", how="inner")
        active_nav = np.cumprod(1.0 + merged["strategy"] - merged["baseline"])
        active_ax.plot(merged["Date"], active_nav, color=color, lw=1.0, label=LABELS[method].replace("EW ", ""))
    active_ax.axhline(1.0, color="#9CA3AF", lw=0.7, zorder=1)
    active_ax.set_title("C  Active NAV of CQL, κ = 0.2", loc="left", fontweight="bold")
    active_ax.set_xlabel("Date")
    active_ax.set_ylabel("Active NAV")
    active_ax.legend(fontsize=5.2, loc="upper left")

    for ax in [nav_ax, scatter_ax, active_ax]:
        ax.tick_params(labelsize=6)
        ax.grid(axis="y", color="#E5E7EB", lw=0.5, zorder=0)

    stem = output_dir / "full_oos_public_replication"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir / "figures"
    stem = make_figure(args.results_dir, output_dir)
    print(f"Wrote {stem}.{{svg,pdf,tiff,png}}")


if __name__ == "__main__":
    main()
