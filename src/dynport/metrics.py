from __future__ import annotations
import math
import numpy as np
import pandas as pd


def performance_metrics(net_returns, gross_returns=None, turnover=None):
    net = np.asarray(list(net_returns), dtype=float)
    gross = np.asarray(list(gross_returns), dtype=float) if gross_returns is not None else net
    turn = np.asarray(list(turnover), dtype=float) if turnover is not None else np.zeros_like(net)
    if len(net) == 0:
        return {k: np.nan for k in ["net_nav", "ann_return", "ann_vol", "net_sharpe", "gross_sharpe", "max_drawdown", "calmar", "avg_turnover", "total_turnover"]} | {"n_days": 0}
    nav = np.cumprod(1 + net)
    ann_return = (1 + net.mean()) ** 252 - 1
    ann_vol = net.std(ddof=1) * math.sqrt(252) if len(net) > 1 else np.nan
    net_sharpe = net.mean() / (net.std(ddof=1) + 1e-12) * math.sqrt(252) if len(net) > 1 else np.nan
    gross_sharpe = gross.mean() / (gross.std(ddof=1) + 1e-12) * math.sqrt(252) if len(gross) > 1 else np.nan
    dd = (nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)
    mdd = float(dd.min())
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else np.nan
    return {"net_nav": float(nav[-1]), "ann_return": float(ann_return), "ann_vol": float(ann_vol), "net_sharpe": float(net_sharpe), "gross_sharpe": float(gross_sharpe), "max_drawdown": mdd, "calmar": calmar, "avg_turnover": float(turn.mean()), "total_turnover": float(turn.sum()), "n_days": int(len(net))}


def add_nav(df: pd.DataFrame, return_col: str = "net_return") -> pd.DataFrame:
    out = df.copy()
    out["nav"] = (1 + out[return_col]).cumprod()
    return out
