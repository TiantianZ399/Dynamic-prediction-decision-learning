from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd


def _parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _long_price_to_returns(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp.columns = ["RIC", "Date", "Price"]
    tmp["Date"] = _parse_date(tmp["Date"])
    tmp["Price"] = pd.to_numeric(tmp["Price"], errors="coerce")
    tmp = tmp.dropna(subset=["RIC", "Date", "Price"]).sort_values(["RIC", "Date"])
    tmp["Return"] = tmp.groupby("RIC")["Price"].pct_change(fill_method=None)
    tmp = tmp.dropna(subset=["Return"])
    wide = tmp.pivot(index="Date", columns="RIC", values="Return").sort_index().fillna(0.0).reset_index()
    wide.columns.name = None
    return wide


def load_returns(path: str | Path, start_date: str | None = None, end_date: str | None = None):
    """Load wide returns or long price CSV.

    Wide format: Date,Asset1,Asset2,...
    Long format: RIC,Date,Price, either headered or headerless.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    if {"ric", "date", "price"}.issubset(lower):
        df = _long_price_to_returns(df[[lower["ric"], lower["date"], lower["price"]]])
    elif "date" in lower:
        if lower["date"] != "Date":
            df = df.rename(columns={lower["date"]: "Date"})
        df["Date"] = _parse_date(df["Date"])
    else:
        df = pd.read_csv(path, header=None, usecols=[0, 1, 2])
        df = _long_price_to_returns(df)
    df["Date"] = _parse_date(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if start_date is not None:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["Date"] <= pd.Timestamp(end_date)]
    assets = [c for c in df.columns if c != "Date"]
    returns = df[assets].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float64)
    return df["Date"].to_numpy(), assets, returns


def save_wide_returns(path: str | Path, dates: Sequence, assets: Sequence[str], returns: np.ndarray) -> None:
    out = pd.DataFrame(returns, columns=list(assets))
    out.insert(0, "Date", pd.to_datetime(dates))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def standardize_by_train(x: np.ndarray, train_idx: np.ndarray):
    mu = x[train_idx].mean(axis=0, keepdims=True)
    sd = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd, mu.squeeze(), sd.squeeze()
