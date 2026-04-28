from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

ACTION_NAMES = ["hold_prev", "equal_weight", "inverse_vol", "inverse_var", "bond_basket", "equity_basket", "gold", "commodity", "top2_momentum", "top3_momentum", "soft_momentum"]

@dataclass(frozen=True)
class ActionConfig:
    warmup: int = 120
    vol_window: int = 20
    momentum_window: int = 60


def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 1e-12 else np.ones_like(w) / len(w)


def cumulative_return(x: np.ndarray) -> np.ndarray:
    return np.prod(1.0 + x, axis=0) - 1.0


def make_candidate_actions(returns: np.ndarray, asset_names: Sequence[str], cfg: ActionConfig = ActionConfig()):
    r = np.asarray(returns, dtype=float)
    T, N = r.shape
    idx = {name: i for i, name in enumerate(asset_names)}
    eq = np.ones(N) / N
    base = np.zeros((T, len(ACTION_NAMES) - 1, N), dtype=float)
    covs = np.zeros((T, N, N), dtype=float)
    for t in range(T):
        if t < cfg.warmup:
            base[t] = np.tile(eq, (len(ACTION_NAMES) - 1, 1))
            covs[t] = np.eye(N) * 1e-6
            continue
        vol = r[max(0, t - cfg.vol_window + 1):t + 1].std(axis=0, ddof=0) + 1e-6
        mom = cumulative_return(r[max(0, t - cfg.momentum_window + 1):t + 1]) / vol
        covs[t] = np.cov(r[max(0, t - cfg.vol_window + 1):t + 1].T, bias=True) + np.eye(N) * 1e-6
        inv_vol = normalize_weights(1 / vol)
        inv_var = normalize_weights(1 / (vol * vol))
        bond = np.zeros(N)
        for n in ["AGG.P", "LQD.P", "MUB.P"]:
            if n in idx: bond[idx[n]] = 1
        equity = np.zeros(N)
        for n in ["VTI.P", "IWM.P"]:
            if n in idx: equity[idx[n]] = 1
        gold = np.zeros(N); gold[idx.get("GLD.P", 0)] = 1
        commodity = np.zeros(N); commodity[idx.get("DBC.P", 0)] = 1
        order = np.argsort(-mom)
        top2 = np.zeros(N); top2[order[:min(2, N)]] = 1 / min(2, N)
        top3 = np.zeros(N); top3[order[:min(3, N)]] = 1 / min(3, N)
        z = np.clip(mom, -5, 5)
        soft = normalize_weights(np.exp(z - z.max()))
        base[t] = np.stack([eq, inv_vol, inv_var, normalize_weights(bond), normalize_weights(equity), gold, commodity, top2, top3, soft])
    return base, covs


def action_weight(action_id: int, time_index: int, previous_weight: np.ndarray, base_actions: np.ndarray) -> np.ndarray:
    return previous_weight.copy() if action_id == 0 else base_actions[time_index, action_id - 1].copy()
