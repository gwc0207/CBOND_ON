from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.factors.defs._intraday_utils import EPS, OPEN_LIKE_CACHE_COL


def rolling_last_rank_pct(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of current value in each rolling window (average tie method)."""
    w = max(1, int(window))
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
    n = int(arr.size)
    out = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return pd.Series(out, index=series.index, dtype="float64")

    if n >= w:
        windows = np.lib.stride_tricks.sliding_window_view(arr, w)
        last = windows[:, -1]
        valid = ~np.isnan(windows)
        valid_count = valid.sum(axis=1)
        last_valid = ~np.isnan(last)
        lt = np.where(valid, windows < last[:, None], False).sum(axis=1)
        eq = np.where(valid, windows == last[:, None], False).sum(axis=1)
        rank = np.full(windows.shape[0], np.nan, dtype="float64")
        ok = last_valid & (valid_count > 0)
        rank[ok] = (lt[ok] + (eq[ok] + 1.0) / 2.0) / valid_count[ok]
        out[w - 1 :] = rank

    prefix_end = min(w - 1, n)
    for i in range(prefix_end):
        window_arr = arr[: i + 1]
        last = window_arr[-1]
        if np.isnan(last):
            continue
        valid = window_arr[~np.isnan(window_arr)]
        if valid.size == 0:
            continue
        lt = np.sum(valid < last)
        eq = np.sum(valid == last)
        out[i] = (lt + (eq + 1.0) / 2.0) / valid.size

    return pd.Series(out, index=series.index, dtype="float64")


def rolling_linear_decay(series: pd.Series, window: int) -> pd.Series:
    w = max(1, int(window))

    def _decay(arr: np.ndarray) -> float:
        n = arr.shape[0]
        if n <= 0:
            return np.nan
        weights = np.arange(1, n + 1, dtype="float64")
        denom = float(weights.sum())
        if denom <= 0.0:
            return np.nan
        return float(np.nansum(arr * weights) / (denom + EPS))

    return pd.to_numeric(series, errors="coerce").rolling(w, min_periods=1).apply(_decay, raw=True)


def mid_price1_series(g: pd.DataFrame) -> pd.Series:
    ask = pd.to_numeric(g["ask_price1"], errors="coerce") if "ask_price1" in g.columns else None
    bid = pd.to_numeric(g["bid_price1"], errors="coerce") if "bid_price1" in g.columns else None
    if ask is not None and bid is not None:
        mid = (ask + bid) / 2.0
        if mid.notna().any():
            return mid
    if OPEN_LIKE_CACHE_COL in g.columns:
        return pd.to_numeric(g[OPEN_LIKE_CACHE_COL], errors="coerce")
    if "open" in g.columns:
        return pd.to_numeric(g["open"], errors="coerce")
    return pd.Series(np.nan, index=g.index, dtype="float64")

