from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from .actions import ACTION_NAMES, ActionConfig, action_weight, make_candidate_actions
from .data import load_returns, standardize_by_train
from .metrics import add_nav, performance_metrics

@dataclass
class FQIConfig:
    start_date: str = "2019-01-02"
    end_date: str = "2024-12-31"
    train_len: int = 250
    test_len: int = 20
    warmup: int = 120
    vol_window: int = 20
    eval_cost: float = 0.002
    reward_cost: float = 0.002
    risk_coef: float = 0.25
    fqi_iters: int = 3
    ridge_alpha: float = 10.0
    gamma_values: tuple[float, ...] = (0.0, 0.95)
    run_extra_trees: bool = False
    et_n_estimators: int = 50
    et_max_depth: int = 5
    max_rolls: int | None = 12
    seed: int = 2020

class FastRidgeMultiOutput:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.mu = None
        self.sd = None
        self.coef = None
    def fit(self, x, y):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        self.mu = x.mean(axis=0, keepdims=True)
        self.sd = x.std(axis=0, keepdims=True) + 1e-8
        xs = (x - self.mu) / self.sd
        xb = np.concatenate([xs, np.ones((xs.shape[0], 1))], axis=1)
        A = xb.T @ xb
        reg = self.alpha * np.eye(A.shape[0]); reg[-1, -1] = 0
        self.coef = np.linalg.solve(A + reg, xb.T @ y)
        return self
    def predict(self, x):
        x = np.asarray(x, dtype=float)
        xs = (x - self.mu) / self.sd
        xb = np.concatenate([xs, np.ones((xs.shape[0], 1))], axis=1)
        return xb @ self.coef

def make_features(returns: np.ndarray, warmup: int = 120, vol_window: int = 20) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    T, N = r.shape
    feats = np.zeros((T, 6 * N), dtype=float)
    for t in range(T):
        if t < warmup: continue
        r1 = r[t]
        r5 = np.prod(1 + r[max(0, t - 4):t + 1], axis=0) - 1
        r20 = np.prod(1 + r[max(0, t - 19):t + 1], axis=0) - 1
        r60 = np.prod(1 + r[max(0, t - 59):t + 1], axis=0) - 1
        vol = r[max(0, t - vol_window + 1):t + 1].std(axis=0, ddof=0) + 1e-6
        mom_vol = r60 / vol
        mom_z = (mom_vol - mom_vol.mean()) / (mom_vol.std() + 1e-6)
        feats[t] = np.r_[r1, r5, r20, r60, vol, mom_z]
    return feats

def build_fqi_training_data(returns, features, base_actions, covariances, train_indices, cfg: FQIConfig):
    N = returns.shape[1]; K = len(ACTION_NAMES); D = features.shape[1] + N
    x_rows, rewards, next_states = [], [], []
    equal = np.ones(N) / N
    for t in train_indices:
        if t < 1 or t + 1 >= len(returns): continue
        prev_candidates = np.vstack([equal, base_actions[t - 1]])[:K]
        for prev in prev_candidates:
            x_rows.append(np.r_[features[t], prev])
            y = np.zeros(K); ns = np.zeros((K, D)); realized = returns[t + 1]; cov = covariances[t]
            for a in range(K):
                w = action_weight(a, t, prev, base_actions)
                turnover = np.abs(w - prev).sum()
                y[a] = float(w @ realized - cfg.reward_cost * turnover - cfg.risk_coef * (w @ cov @ w))
                ns[a] = np.r_[features[t + 1], w]
            rewards.append(y); next_states.append(ns)
    return np.asarray(x_rows), np.asarray(rewards), np.asarray(next_states)

def _make_regressor(kind: str, cfg: FQIConfig):
    if kind == "ridge": return FastRidgeMultiOutput(alpha=cfg.ridge_alpha)
    if kind == "extra_trees": return ExtraTreesRegressor(n_estimators=cfg.et_n_estimators, max_depth=cfg.et_max_depth, min_samples_leaf=8, random_state=cfg.seed, n_jobs=1)
    raise ValueError(kind)

def fit_fqi(x, rewards, next_states, gamma: float, kind: str, cfg: FQIConfig):
    K = rewards.shape[1]
    y = np.clip(rewards, -0.05, 0.05)
    next_flat = next_states.reshape(-1, x.shape[1])
    model = None
    for _ in range(cfg.fqi_iters):
        if model is not None and gamma > 0:
            q_next = model.predict(next_flat).reshape(len(x), K, K)
            y = np.clip(rewards + gamma * q_next.max(axis=2), -0.05, 0.05)
        model = _make_regressor(kind, cfg)
        model.fit(x, y)
    return model

def evaluate_policy(model, method_name, dates, returns, features, base_actions, start: int, end: int, cfg: FQIConfig):
    N = returns.shape[1]
    prev = np.ones(N) / N
    rows = []
    counts = {name: 0 for name in ACTION_NAMES}
    for t in range(start, min(end, len(returns) - 1)):
        if method_name == "equal_weight":
            action = 1; w = np.ones(N) / N
        else:
            q = model.predict(np.r_[features[t], prev].reshape(1, -1)).reshape(-1)
            action = int(np.argmax(q)); w = action_weight(action, t, prev, base_actions)
        gross = float(w @ returns[t + 1]); turnover = float(np.abs(w - prev).sum())
        rows.append({"Date": pd.Timestamp(dates[t + 1]).date().isoformat(), "method": method_name, "gross_return": gross, "net_return": gross - cfg.eval_cost * turnover, "turnover": turnover, "action": ACTION_NAMES[action]})
        counts[ACTION_NAMES[action]] += 1; prev = w.copy()
    return add_nav(pd.DataFrame(rows)), counts

def rolling_fqi(data_path: str | Path, out_dir: str | Path, cfg: FQIConfig):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    dates, assets, returns = load_returns(data_path, start_date="2011-01-01", end_date=cfg.end_date)
    ds = pd.Series(pd.to_datetime(dates))
    raw_features = make_features(returns, cfg.warmup, cfg.vol_window)
    base_actions, covariances = make_candidate_actions(returns, assets, ActionConfig(cfg.warmup, cfg.vol_window))
    start_idx = int(np.where(ds >= pd.Timestamp(cfg.start_date))[0][0])
    end_idx = int(np.where(ds <= pd.Timestamp(cfg.end_date))[0][-1]) - 1
    roll_starts = list(range(start_idx, end_idx - cfg.test_len + 1, cfg.test_len))
    if cfg.max_rolls is not None: roll_starts = roll_starts[:cfg.max_rolls]
    methods = ["equal_weight"] + [f"fqi_ridge_g{g:g}" for g in cfg.gamma_values]
    if cfg.run_extra_trees: methods.append("fqi_extra_trees_g0.95")
    curves = {m: [] for m in methods}; action_counts = {m: {a: 0 for a in ACTION_NAMES} for m in methods}; roll_rows = []
    for roll, start in enumerate(roll_starts):
        train_start, train_end = start - cfg.train_len - 1, start - 2
        if train_start < cfg.warmup or train_end <= train_start: continue
        train_idx = np.arange(train_start, train_end + 1)
        features, _, _ = standardize_by_train(raw_features, train_idx)
        x, rewards, next_states = build_fqi_training_data(returns, features, base_actions, covariances, train_idx, cfg)
        te0, te1 = start, min(start + cfg.test_len, len(returns) - 1)
        for method, model in [("equal_weight", None)]:
            df, counts = evaluate_policy(model, method, dates, returns, features, base_actions, te0, te1, cfg)
            curves[method].append(df)
            for k, v in counts.items(): action_counts[method][k] += v
            met = performance_metrics(df.net_return, df.gross_return, df.turnover); met.update({"roll": roll, "method": method}); roll_rows.append(met)
        for gamma in cfg.gamma_values:
            model = fit_fqi(x, rewards, next_states, gamma, "ridge", cfg); method = f"fqi_ridge_g{gamma:g}"
            df, counts = evaluate_policy(model, method, dates, returns, features, base_actions, te0, te1, cfg)
            curves[method].append(df)
            for k, v in counts.items(): action_counts[method][k] += v
            met = performance_metrics(df.net_return, df.gross_return, df.turnover); met.update({"roll": roll, "method": method}); roll_rows.append(met)
        if cfg.run_extra_trees:
            model = fit_fqi(x, rewards, next_states, 0.95, "extra_trees", cfg); method = "fqi_extra_trees_g0.95"
            df, counts = evaluate_policy(model, method, dates, returns, features, base_actions, te0, te1, cfg)
            curves[method].append(df)
            for k, v in counts.items(): action_counts[method][k] += v
            met = performance_metrics(df.net_return, df.gross_return, df.turnover); met.update({"roll": roll, "method": method}); roll_rows.append(met)
    agg_rows = []
    for method, segs in curves.items():
        if not segs: continue
        df = add_nav(pd.concat(segs, ignore_index=True)); df.to_csv(out_dir / f"{method}_daily_curve.csv", index=False)
        met = performance_metrics(df.net_return, df.gross_return, df.turnover); met["method"] = method; agg_rows.append(met)
    agg = pd.DataFrame(agg_rows).sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    agg.to_csv(out_dir / "aggregate_results.csv", index=False)
    pd.DataFrame(roll_rows).to_csv(out_dir / "roll_level_results.csv", index=False)
    pd.DataFrame([{"method": m, **c} for m, c in action_counts.items()]).to_csv(out_dir / "action_counts.csv", index=False)
    pd.Series({"assets": assets, "config": asdict(cfg), "action_names": ACTION_NAMES}).to_json(out_dir / "metadata.json", indent=2)
    return agg
