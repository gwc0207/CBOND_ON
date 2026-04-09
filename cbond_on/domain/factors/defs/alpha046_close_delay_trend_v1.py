from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha046_close_delay_trend_v1")
class Alpha046CloseDelayTrendV1Factor(_AlphaBase):
    name = "alpha046_close_delay_trend_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delay_window_long = int(ctx.params.get("delay_window_long", 10))
        delay_window_short = int(ctx.params.get("delay_window_short", 5))
        trend_scale = float(ctx.params.get("trend_scale", 10.0))
        threshold_up = float(ctx.params.get("threshold_up", 0.25))
        threshold_down = float(ctx.params.get("threshold_down", 0.0))
        frame = _prepare_panel(ctx, ["last"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            d_long = last_px.shift(max(1, delay_window_long))
            d_short = last_px.shift(max(1, delay_window_short))
            diff1 = (d_long - d_short) / trend_scale
            diff2 = (d_short - last_px) / trend_scale
            trend = float((diff1 - diff2).iloc[-1]) if pd.notna((diff1 - diff2).iloc[-1]) else 0.0
            delta_close = float(last_px.diff(1).iloc[-1]) if pd.notna(last_px.diff(1).iloc[-1]) else 0.0
            if trend > threshold_up:
                return -1.0
            if trend < threshold_down:
                return 1.0
            return float(-delta_close)

        return _group_scalar(frame, _calc)


