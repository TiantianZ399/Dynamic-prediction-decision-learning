
from __future__ import annotations
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .actions import ACTION_NAMES, ActionConfig, action_weight, make_candidate_actions
from .data import load_returns
from .metrics import add_nav, performance_metrics


@dataclass
class DeepTemporalCQLConfig:
    """Config for deep temporal CQL-FQI.

    The model uses a temporal deep encoder (TCN, CNN-LSTM, or MLP) to encode
    past returns and a finite-action conservative Q head to choose a portfolio
    candidate under transaction-cost and risk-aware rewards.
    """
    start_date: str = "2019-01-02"
    end_date: str = "2024-12-31"
    train_len: int = 250
    test_len: int = 20
    warmup: int = 120
    seq_len: int = 120
    vol_window: int = 20
    eval_cost: float = 0.002
    reward_cost: float = 0.002
    risk_coef: float = 0.25
    gamma: float = 0.95
    fqi_iters: int = 2
    epochs_per_iter: int = 3
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-4
    q_scale: float = 100.0
    pred_loss_weight: float = 0.05
    cql_alpha: float = 0.05
    hidden: int = 32
    dropout: float = 0.10
    encoder: str = "tcn"  # tcn | cnn_lstm | mlp
    kernel_size: int = 5
    lstm_layers: int = 1
    max_rolls: int | None = 12
    switch_thresholds: tuple[float, ...] = (0.0, 0.10, 0.20)
    seed: int = 2021
    torch_num_threads: int = 1


def _standardize_returns(returns: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mu = returns[train_idx].mean(axis=0, keepdims=True)
    sd = returns[train_idx].std(axis=0, keepdims=True) + 1e-6
    return (returns - mu) / sd


def _hist(Rs: np.ndarray, t: int, cfg: DeepTemporalCQLConfig) -> np.ndarray:
    return Rs[t - cfg.seq_len + 1 : t + 1].astype(np.float32)


def build_transition_data(returns, standardized_returns, base_actions, covariances, train_indices, cfg):
    N = returns.shape[1]
    K = len(ACTION_NAMES)
    eq = np.ones(N) / N
    H, P, Y, NH, NP, PT = [], [], [], [], [], []
    for t in train_indices:
        if t < cfg.seq_len or t + 1 >= len(returns):
            continue
        ht = _hist(standardized_returns, t, cfg)
        hn = _hist(standardized_returns, t + 1, cfg)
        realized = returns[t + 1]
        cov = covariances[t]
        prev_candidates = np.vstack([eq, base_actions[t - 1]])[:K]
        for prev in prev_candidates:
            y = np.zeros(K, dtype=np.float32)
            next_prev = np.zeros((K, N), dtype=np.float32)
            for a in range(K):
                w = action_weight(a, t, prev, base_actions)
                y[a] = float(
                    w @ realized
                    - cfg.reward_cost * np.abs(w - prev).sum()
                    - cfg.risk_coef * (w @ cov @ w)
                )
                next_prev[a] = w.astype(np.float32)
            H.append(ht)
            P.append(prev.astype(np.float32))
            Y.append(y * cfg.q_scale)
            NH.append(np.repeat(hn.reshape(1, cfg.seq_len, N), K, axis=0))
            NP.append(next_prev)
            PT.append((realized * cfg.q_scale).astype(np.float32))
    return {
        "hist": np.asarray(H, np.float32),
        "prev": np.asarray(P, np.float32),
        "rewards": np.asarray(Y, np.float32),
        "next_hist": np.asarray(NH, np.float32),
        "next_prev": np.asarray(NP, np.float32),
        "pred_target": np.asarray(PT, np.float32),
    }


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad > 0:
            y = y[..., :-self.pad]
        return y


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x))
        y = self.dropout(y)
        y = F.relu(self.conv2(y))
        y = self.dropout(y)
        y = y + self.res(x)
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class TCNEncoder(nn.Module):
    def __init__(self, n_assets: int, hidden: int, kernel_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(n_assets, hidden, kernel_size, dilation=1, dropout=dropout),
            TCNBlock(hidden, hidden, kernel_size, dilation=2, dropout=dropout),
            TCNBlock(hidden, hidden, kernel_size, dilation=4, dropout=dropout),
        )

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        z = self.net(hist.transpose(1, 2))
        return z[:, :, -1]


class CNNLSTMEncoder(nn.Module):
    def __init__(self, n_assets: int, hidden: int, kernel_size: int, dropout: float, lstm_layers: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_assets, hidden, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        x = self.conv(hist.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)
        return out[:, -1, :]


class MLPEncoder(nn.Module):
    def __init__(self, seq_len: int, n_assets: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * n_assets, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return self.net(hist.flatten(start_dim=1))


class DeepTemporalQNetwork(nn.Module):
    def __init__(self, cfg: DeepTemporalCQLConfig, n_assets: int, n_actions: int):
        super().__init__()
        enc = cfg.encoder.lower().replace("-", "_")
        if enc == "tcn":
            self.encoder = TCNEncoder(n_assets, cfg.hidden, cfg.kernel_size, cfg.dropout)
        elif enc in {"cnn_lstm", "cnnlstm"}:
            self.encoder = CNNLSTMEncoder(n_assets, cfg.hidden, cfg.kernel_size, cfg.dropout, cfg.lstm_layers)
        elif enc == "mlp":
            self.encoder = MLPEncoder(cfg.seq_len, n_assets, cfg.hidden, cfg.dropout)
        else:
            raise ValueError(f"Unknown encoder: {cfg.encoder}")
        self.trunk = nn.Sequential(
            nn.Linear(cfg.hidden + n_assets, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.q_head = nn.Linear(cfg.hidden, n_actions)
        self.pred_head = nn.Linear(cfg.hidden, n_assets)

    def forward(self, hist: torch.Tensor, prev: torch.Tensor):
        z_hist = self.encoder(hist)
        z = self.trunk(torch.cat([z_hist, prev], dim=-1))
        return self.q_head(z), self.pred_head(z)


def train_model(data, cfg: DeepTemporalCQLConfig, n_assets: int):
    K = len(ACTION_NAMES)
    if len(data["hist"]) == 0:
        raise ValueError("No transitions. Increase data length or reduce seq_len/warmup.")
    model = DeepTemporalQNetwork(cfg, n_assets, K)
    target = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    H = torch.from_numpy(data["hist"])
    P = torch.from_numpy(data["prev"])
    R = torch.from_numpy(data["rewards"])
    NH = torch.from_numpy(data["next_hist"])
    NP = torch.from_numpy(data["next_prev"])
    PT = torch.from_numpy(data["pred_target"])
    M = H.shape[0]
    y = torch.clamp(R.clone(), -5, 5)
    for it in range(cfg.fqi_iters):
        if it > 0 and cfg.gamma > 0:
            flatH = NH.reshape(M * K, cfg.seq_len, n_assets)
            flatP = NP.reshape(M * K, n_assets)
            vals = []
            target.eval()
            with torch.no_grad():
                for st in range(0, M * K, 8192):
                    q, _ = target(flatH[st : st + 8192], flatP[st : st + 8192])
                    vals.append(q.max(1).values)
                y = torch.clamp(R + cfg.gamma * torch.cat(vals).reshape(M, K), -5, 5)
        dl = DataLoader(TensorDataset(H, P, y, PT), batch_size=cfg.batch_size, shuffle=True)
        model.train()
        for _ in range(cfg.epochs_per_iter):
            for bh, bp, by, bt in dl:
                opt.zero_grad()
                q, pred = model(bh, bp)
                loss_q = F.smooth_l1_loss(q, by)
                loss_p = F.smooth_l1_loss(pred, bt)
                cql = (torch.logsumexp(q, dim=1) - q[:, 0]).mean()
                loss = loss_q + cfg.pred_loss_weight * loss_p + cfg.cql_alpha * cql
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
        target.load_state_dict(model.state_dict())
    return model


def eval_equal(dates, returns, start, end, cfg):
    N = returns.shape[1]
    w = np.ones(N) / N
    prev = w.copy()
    rows = []
    for t in range(start, min(end, len(returns) - 1)):
        gross = float(w @ returns[t + 1])
        turnover = float(np.abs(w - prev).sum())
        rows.append({
            "Date": pd.Timestamp(dates[t + 1]).date().isoformat(),
            "method": "equal_weight",
            "gross_return": gross,
            "net_return": gross - cfg.eval_cost * turnover,
            "turnover": turnover,
            "action": "equal_weight",
        })
        prev = w.copy()
    return add_nav(pd.DataFrame(rows))


def eval_model(model, method, dates, returns, standardized_returns, base_actions, start, end, cfg, threshold):
    N = returns.shape[1]
    prev = np.ones(N) / N
    rows = []
    counts = {a: 0 for a in ACTION_NAMES}
    model.eval()
    with torch.no_grad():
        for t in range(start, min(end, len(returns) - 1)):
            h = torch.from_numpy(_hist(standardized_returns, t, cfg).reshape(1, cfg.seq_len, N))
            p = torch.from_numpy(prev.reshape(1, N).astype(np.float32))
            q, _ = model(h, p)
            qv = q.squeeze(0).cpu().numpy()
            best = int(np.argmax(qv))
            action = best if qv[best] - qv[0] > threshold else 0
            w = action_weight(action, t, prev, base_actions)
            gross = float(w @ returns[t + 1])
            turnover = float(np.abs(w - prev).sum())
            rows.append({
                "Date": pd.Timestamp(dates[t + 1]).date().isoformat(),
                "method": method,
                "gross_return": gross,
                "net_return": gross - cfg.eval_cost * turnover,
                "turnover": turnover,
                "action": ACTION_NAMES[action],
                "q_hold": float(qv[0]),
                "q_best": float(qv[best]),
                "best_action": ACTION_NAMES[best],
            })
            counts[ACTION_NAMES[action]] += 1
            prev = w.copy()
    return add_nav(pd.DataFrame(rows)), counts


def rolling_deep_temporal_cql_fqi(data_path: str | Path, out_dir: str | Path, cfg: DeepTemporalCQLConfig):
    torch.set_num_threads(max(1, int(cfg.torch_num_threads)))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates, assets, returns = load_returns(data_path, start_date="2011-01-01", end_date=cfg.end_date)
    ds = pd.Series(pd.to_datetime(dates))
    base_actions, covariances = make_candidate_actions(returns, assets, ActionConfig(cfg.warmup, cfg.vol_window))
    start_idx = int(np.where(ds >= pd.Timestamp(cfg.start_date))[0][0])
    end_idx = int(np.where(ds <= pd.Timestamp(cfg.end_date))[0][-1]) - 1
    roll_starts = list(range(start_idx, end_idx - cfg.test_len + 1, cfg.test_len))
    if cfg.max_rolls is not None:
        roll_starts = roll_starts[: cfg.max_rolls]
    methods = ["equal_weight"] + [f"{cfg.encoder}_cql_thr{thr:g}" for thr in cfg.switch_thresholds]
    curves = {m: [] for m in methods}
    actions = {m: {a: 0 for a in ACTION_NAMES} for m in methods}
    roll_rows = []
    for roll, start in enumerate(roll_starts):
        train_idx = np.arange(start - cfg.train_len - 1, start - 1)
        train_idx = train_idx[(train_idx >= cfg.warmup) & (train_idx < len(returns) - 1)]
        if len(train_idx) == 0:
            continue
        Rs = _standardize_returns(returns, train_idx)
        data = build_transition_data(returns, Rs, base_actions, covariances, train_idx, cfg)
        te0, te1 = start, min(start + cfg.test_len, len(returns) - 1)
        df_eq = eval_equal(dates, returns, te0, te1, cfg)
        curves["equal_weight"].append(df_eq)
        met = performance_metrics(df_eq.net_return, df_eq.gross_return, df_eq.turnover)
        met.update({"roll": roll, "method": "equal_weight", "encoder": cfg.encoder})
        roll_rows.append(met)
        model = train_model(data, cfg, returns.shape[1])
        for threshold in cfg.switch_thresholds:
            method = f"{cfg.encoder}_cql_thr{threshold:g}"
            df, counts = eval_model(model, method, dates, returns, Rs, base_actions, te0, te1, cfg, threshold)
            curves[method].append(df)
            for k, v in counts.items():
                actions[method][k] += v
            met = performance_metrics(df.net_return, df.gross_return, df.turnover)
            met.update({"roll": roll, "method": method, "encoder": cfg.encoder})
            roll_rows.append(met)
    agg_rows = []
    for method, segs in curves.items():
        if not segs:
            continue
        df = add_nav(pd.concat(segs, ignore_index=True))
        df.to_csv(out_dir / f"{method}_daily_curve.csv", index=False)
        met = performance_metrics(df.net_return, df.gross_return, df.turnover)
        met["method"] = method
        met["encoder"] = cfg.encoder
        agg_rows.append(met)
    agg = pd.DataFrame(agg_rows).sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    agg.to_csv(out_dir / "aggregate_results.csv", index=False)
    pd.DataFrame(roll_rows).to_csv(out_dir / "roll_level_results.csv", index=False)
    pd.DataFrame([{"method": m, **c} for m, c in actions.items()]).to_csv(out_dir / "action_counts.csv", index=False)
    pd.Series({"assets": assets, "config": asdict(cfg), "action_names": ACTION_NAMES}).to_json(out_dir / "metadata.json", indent=2)
    return agg
