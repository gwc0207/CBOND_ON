from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel, _ts_rank_last


@FactorRegistry.register("alpha052_low_momentum_volume_v1")
class Alpha052LowMomentumVolumeV1Factor(_AlphaBase):
    name = "alpha052_low_momentum_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_min_window = int(ctx.params.get("ts_min_window", 5))
        delay_window = int(ctx.params.get("delay_window", 5))
        sum_window_long = int(ctx.params.get("sum_window_long", 60))
        sum_window_short = int(ctx.params.get("sum_window_short", 20))
        ts_rank_window = int(ctx.params.get("ts_rank_window", 5))
        frame = _prepare_panel(ctx, ["low", "volume", "last", "pre_close"])

        def _calc(g: pd.DataFrame) -> float:
            low = g["low"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            pre_close = g["pre_close"].astype("float64")

            ts_min_low = low.rolling(max(1, ts_min_window), min_periods=1).min()
            delay_min = ts_min_low.shift(max(1, delay_window))
            low_diff = (-ts_min_low) + delay_min

            returns = (last_px - pre_close) / (pre_close + EPS)
            sum_ret_long = returns.rolling(max(1, sum_window_long), min_periods=1).sum()
            sum_ret_short = returns.rolling(max(1, sum_window_short), min_periods=1).sum()
            denom = float(max(1, sum_window_long - sum_window_short))
            ret_diff = (sum_ret_long - sum_ret_short) / denom
            rank_ret = ret_diff.rank(pct=True, method="average")
            ts_rank_vol = _ts_rank_last(volume, ts_rank_window)
            alpha = low_diff * rank_ret * ts_rank_vol
            val = alpha.iloc[-1]
            if pd.isna(val):
                return 0.0
            return float(val)

        return _group_scalar(frame, _calc)

