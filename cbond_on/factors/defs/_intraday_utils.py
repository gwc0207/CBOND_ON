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
