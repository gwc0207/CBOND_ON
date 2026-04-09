from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _corr_last, _cs_rank, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha045_close_sum_corr_v1")
class Alpha045CloseSumCorrV1Factor(_AlphaBase):
    name = "alpha045_close_sum_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delay_window = int(ctx.params.get("delay_window", 5))
        sum_window_long = int(ctx.params.get("sum_window_long", 20))
        corr_window_1 = int(ctx.params.get("corr_window_1", 2))
        sum_window_short = int(ctx.params.get("sum_window_short", 5))
        corr_window_2 = int(ctx.params.get("corr_window_2", 2))
        frame = _prepare_panel(ctx, ["last", "volume"])

        def _avg_delay(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            delayed = last_px.shift(max(1, delay_window))
            avg_delay = delayed.rolling(max(1, sum_window_long), min_periods=1).mean()
            return float(avg_delay.iloc[-1])

        def _corr1(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            volume = g["volume"].astype("float64")
            return float(_corr_last(last_px, volume, corr_window_1))

        def _corr2(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            sum_short = last_px.rolling(max(1, sum_window_short), min_periods=1).sum()
            sum_long = last_px.rolling(max(1, sum_window_long), min_periods=1).sum()
            return float(_corr_last(sum_short, sum_long, corr_window_2))

        avg_delay = _group_scalar(frame, _avg_delay)
        corr1 = _group_scalar(frame, _corr1)
        corr2 = _group_scalar(frame, _corr2)
        return -(_cs_rank(avg_delay) * corr1 * _cs_rank(corr2 + EPS))


