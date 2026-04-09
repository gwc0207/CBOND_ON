from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import _AlphaBase, _group_scalar, _prepare_panel


def _rolling_last_rank_pct(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of current value in each rolling window (average tie method)."""
    w = max(1, int(window))
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
    n = int(arr.size)
    out = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return pd.Series(out, index=series.index, dtype="float64")

    full = w
    if n >= full:
        windows = np.lib.stride_tricks.sliding_window_view(arr, full)
        last = windows[:, -1]
        valid = ~np.isnan(windows)
        valid_count = valid.sum(axis=1)
        last_valid = ~np.isnan(last)
        lt = np.where(valid, windows < last[:, None], False).sum(axis=1)
        eq = np.where(valid, windows == last[:, None], False).sum(axis=1)
        rank = np.full(windows.shape[0], np.nan, dtype="float64")
        ok = last_valid & (valid_count > 0)
        rank[ok] = (lt[ok] + (eq[ok] + 1.0) / 2.0) / valid_count[ok]
        out[full - 1 :] = rank

    prefix_end = min(full - 1, n)
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


@FactorRegistry.register("alpha026_volume_high_rank_corr_v1")
class Alpha026VolumeHighRankCorrV1Factor(_AlphaBase):
    name = "alpha026_volume_high_rank_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_rank_window = int(ctx.params.get("ts_rank_window", 5))
        corr_window = int(ctx.params.get("corr_window", 5))
        ts_max_window = int(ctx.params.get("ts_max_window", 3))
        frame = _prepare_panel(ctx, ["volume", "high"])

        def _calc(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64")
            high = g["high"].astype("float64")
            ts_rank_vol = _rolling_last_rank_pct(volume, ts_rank_window)
            ts_rank_high = _rolling_last_rank_pct(high, ts_rank_window)
            corr = ts_rank_vol.rolling(max(2, corr_window), min_periods=2).corr(ts_rank_high)
            ts_max = corr.rolling(max(1, ts_max_window), min_periods=1).max().iloc[-1]
            if pd.isna(ts_max):
                return 0.0
            return float(-ts_max)

        return _group_scalar(frame, _calc)


