from __future__ import annotations

from datetime import time as dt_time
from typing import Callable

import numpy as np
import pandas as pd

from cbond_on.factors.base import ensure_panel_index


def ensure_trade_time(panel: pd.DataFrame) -> pd.DataFrame:
    if "trade_time" not in panel.columns:
        raise KeyError("panel missing column: trade_time")
    if not pd.api.types.is_datetime64_any_dtype(panel["trade_time"]):
        panel = panel.copy()
        panel["trade_time"] = pd.to_datetime(panel["trade_time"], errors="coerce")
    return ensure_panel_index(panel)


def group_apply_scalar(panel: pd.DataFrame, func: Callable[[pd.DataFrame], float]) -> pd.Series:
    grouped = panel.groupby(level=["dt", "code"], sort=False)
    out = grouped.apply(func)
    if isinstance(out.index, pd.MultiIndex) and out.index.nlevels > 2:
        out = out.droplevel(-1)
    return out


def slice_window(df: pd.DataFrame, window_minutes: int | None) -> pd.DataFrame:
    if window_minutes is None or int(window_minutes) <= 0:
        return df
    end_time = df["trade_time"].iloc[-1]
    start_time = end_time - pd.Timedelta(minutes=int(window_minutes))
    return df[df["trade_time"] >= start_time]


def parse_hhmm(value: str) -> dt_time:
    parts = str(value).split(":")
    if len(parts) < 2:
        raise ValueError(f"invalid time value: {value}")
    return dt_time(int(parts[0]), int(parts[1]))


def first_last_price(df: pd.DataFrame, price_col: str) -> tuple[float | None, float | None]:
    if df.empty or price_col not in df.columns:
        return None, None
    series = df[price_col].astype("float64")
    if series.empty:
        return None, None
    return float(series.iloc[0]), float(series.iloc[-1])


def open_like_series(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    ask_col: str = "ask_price1",
    bid_col: str = "bid_price1",
) -> pd.Series:
    """Return open-like price series: prefer mid_price1, fallback to open."""
    mid: pd.Series | None = None
    if ask_col in df.columns and bid_col in df.columns:
        ask = pd.to_numeric(df[ask_col], errors="coerce")
        bid = pd.to_numeric(df[bid_col], errors="coerce")
        mid = (ask + bid) / 2.0

    open_px: pd.Series | None = None
    if open_col in df.columns:
        open_px = pd.to_numeric(df[open_col], errors="coerce")

    if mid is not None and open_px is not None:
        return mid.where(mid.notna(), open_px)
    if mid is not None:
        return mid
    if open_px is not None:
        return open_px
    raise KeyError(
        f"open-like requires [{open_col}] or [{ask_col}, {bid_col}] in panel columns"
    )
