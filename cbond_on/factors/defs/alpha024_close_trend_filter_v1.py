from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import EPS, _AlphaBase, _delta_last, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha024_close_trend_filter_v1")
class Alpha024CloseTrendFilterV1Factor(_AlphaBase):
    name = "alpha024_close_trend_filter_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        sum_window = int(ctx.params.get("sum_window", 20))
        delta_window = int(ctx.params.get("delta_window", 20))
        ts_min_window = int(ctx.params.get("ts_min_window", 20))
        short_delta_window = int(ctx.params.get("short_delta_window", 3))
        trend_threshold = float(ctx.params.get("trend_threshold", 0.05))
        frame = _prepare_panel(ctx, ["last"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            avg_close = last_px.rolling(max(1, sum_window), min_periods=1).mean()
            delta_avg = avg_close.diff(max(1, delta_window))
            base = avg_close.shift(max(1, delta_window))
            rate = float(delta_avg.iloc[-1] / (base.iloc[-1] + EPS)) if pd.notna(base.iloc[-1]) else 0.0
            if rate <= trend_threshold:
                ts_min = float(last_px.rolling(max(1, ts_min_window), min_periods=1).min().iloc[-1])
                return float(-(last_px.iloc[-1] - ts_min))
            return float(-_delta_last(last_px, short_delta_window))

        return _group_scalar(frame, _calc)

