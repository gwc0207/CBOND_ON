from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


def _rolling_argmax_position(series: pd.Series, window: int) -> pd.Series:
    w = max(1, int(window))
    return series.rolling(w, min_periods=1).apply(
        lambda arr: float(np.nanargmax(arr)) + 1.0 if np.any(~np.isnan(arr)) else np.nan,
        raw=True,
    )


def _rolling_linear_decay(series: pd.Series, window: int) -> pd.Series:
    w = max(1, int(window))

    def _decay(arr: np.ndarray) -> float:
        n = arr.shape[0]
        if n <= 0:
            return np.nan
        weights = np.arange(1, n + 1, dtype="float64")
        denom = float(weights.sum())
        if denom <= 0.0:
            return np.nan
        return float(np.nansum(arr * weights) / denom)

    return series.rolling(w, min_periods=1).apply(_decay, raw=True)


@FactorRegistry.register("alpha057_close_vwap_decay_v1")
class Alpha057CloseVwapDecayV1Factor(_AlphaBase):
    name = "alpha057_close_vwap_decay_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_argmax_window = int(ctx.params.get("ts_argmax_window", 10))
        decay_window = int(ctx.params.get("decay_window", 2))
        frame = _prepare_panel(ctx, ["last", "amount", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            vwap = amount / (volume + EPS)
            diff = last_px - vwap
            ts_argmax = _rolling_argmax_position(last_px, ts_argmax_window)
            rank_max = ts_argmax.rank(pct=True, method="average")
            decay = _rolling_linear_decay(rank_max, decay_window)
            alpha = -(diff / (decay + EPS))
            val = alpha.iloc[-1]
            if pd.isna(val):
                return 0.0
            return float(val)

        return _group_scalar(frame, _calc)

