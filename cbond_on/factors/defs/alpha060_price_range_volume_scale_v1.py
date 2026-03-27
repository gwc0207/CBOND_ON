from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _cs_rank, _group_scalar, _prepare_panel


def _cs_scale(series: pd.Series) -> pd.Series:
    if series.empty:
        return series

    def _scale(s: pd.Series) -> pd.Series:
        denom = float(s.abs().sum())
        if denom <= EPS:
            return pd.Series(0.0, index=s.index, dtype="float64")
        return s / denom

    return series.groupby(level="dt", group_keys=False).apply(_scale).astype("float64")


def _argmax_pos_last(series: pd.Series, window: int) -> float:
    w = max(1, int(window))
    tail = pd.to_numeric(series.tail(w), errors="coerce").to_numpy(dtype="float64")
    if tail.size == 0 or not np.any(~np.isnan(tail)):
        return 0.0
    return float(np.nanargmax(tail) + 1.0)


@FactorRegistry.register("alpha060_price_range_volume_scale_v1")
class Alpha060PriceRangeVolumeScaleV1Factor(_AlphaBase):
    name = "alpha060_price_range_volume_scale_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_argmax_window = int(ctx.params.get("ts_argmax_window", 10))
        frame = _prepare_panel(ctx, ["last", "high", "low", "volume"])

        def _pos_vol(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            high = g["high"].astype("float64")
            low = g["low"].astype("float64")
            volume = g["volume"].astype("float64")
            numerator = (last_px - low) - (high - last_px)
            denominator = high - low
            position = numerator / (denominator + EPS)
            return float((position * volume).iloc[-1])

        def _ts_argmax(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            return _argmax_pos_last(last_px, ts_argmax_window)

        pos_vol = _group_scalar(frame, _pos_vol)
        rank_scaled = _cs_rank(_cs_scale(_cs_rank(pos_vol)))
        ts_argmax = _group_scalar(frame, _ts_argmax)
        scale_max = _cs_scale(_cs_rank(ts_argmax))
        return -((2.0 * rank_scaled) - scale_max)

