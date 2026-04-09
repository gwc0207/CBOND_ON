from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _cs_rank, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha030_close_sign_volume_v1")
class Alpha030CloseSignVolumeV1Factor(_AlphaBase):
    name = "alpha030_close_sign_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delay1 = int(ctx.params.get("delay1", 1))
        delay2 = int(ctx.params.get("delay2", 2))
        delay3 = int(ctx.params.get("delay3", 3))
        sum_window_short = int(ctx.params.get("sum_window_short", 5))
        sum_window_long = int(ctx.params.get("sum_window_long", 10))
        frame = _prepare_panel(ctx, ["last", "volume"])

        def _sign_sum(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            s1 = np.sign(last_px.iloc[-1] - last_px.shift(max(1, delay1)).iloc[-1])
            s2 = np.sign(last_px.shift(max(1, delay1)).iloc[-1] - last_px.shift(max(1, delay2)).iloc[-1])
            s3 = np.sign(last_px.shift(max(1, delay2)).iloc[-1] - last_px.shift(max(1, delay3)).iloc[-1])
            return float(np.nan_to_num(s1, nan=0.0) + np.nan_to_num(s2, nan=0.0) + np.nan_to_num(s3, nan=0.0))

        def _vol_ratio(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64")
            sum_short = float(volume.rolling(max(1, sum_window_short), min_periods=1).sum().iloc[-1])
            sum_long = float(volume.rolling(max(1, sum_window_long), min_periods=1).sum().iloc[-1])
            return float(sum_short / (sum_long + EPS))

        sign_sum = _group_scalar(frame, _sign_sum)
        vol_ratio = _group_scalar(frame, _vol_ratio)
        rank_sign = 1.0 - _cs_rank(sign_sum)
        return rank_sign * vol_ratio



