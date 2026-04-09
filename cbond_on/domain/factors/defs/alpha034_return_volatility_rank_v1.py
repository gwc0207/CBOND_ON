from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _cs_rank, _delta_last, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha034_return_volatility_rank_v1")
class Alpha034ReturnVolatilityRankV1Factor(_AlphaBase):
    name = "alpha034_return_volatility_rank_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        stddev_window_short = int(ctx.params.get("stddev_window_short", 2))
        stddev_window_long = int(ctx.params.get("stddev_window_long", 5))
        delta_window = int(ctx.params.get("delta_window", 1))
        frame = _prepare_panel(ctx, ["last", "prev_bar_close"])

        def _vol_ratio(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["prev_bar_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            std_short = float(
                returns.rolling(max(2, stddev_window_short), min_periods=2).std().fillna(0.0).iloc[-1]
            )
            std_long = float(
                returns.rolling(max(2, stddev_window_long), min_periods=2).std().fillna(0.0).iloc[-1]
            )
            return float(std_short / (std_long + EPS))

        def _delta_close(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            return float(_delta_last(last_px, delta_window))

        vol_ratio = _group_scalar(frame, _vol_ratio)
        delta_close = _group_scalar(frame, _delta_close)
        rank_vol = 1.0 - _cs_rank(vol_ratio)
        rank_delta = 1.0 - _cs_rank(delta_close)
        return rank_vol + rank_delta




