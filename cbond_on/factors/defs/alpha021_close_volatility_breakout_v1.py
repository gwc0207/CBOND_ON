from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import (
    EPS,
    _AlphaBase,
    _group_scalar,
    _prepare_panel,
)


@FactorRegistry.register("alpha021_close_volatility_breakout_v1")
class Alpha021CloseVolatilityBreakoutV1Factor(_AlphaBase):
    name = "alpha021_close_volatility_breakout_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        sum_window_long = int(ctx.params.get("sum_window_long", 5))
        sum_window_short = int(ctx.params.get("sum_window_short", 2))
        adv_window = int(ctx.params.get("adv_window", 10))
        frame = _prepare_panel(ctx, ["last", "volume", "amount"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            volume = g["volume"].astype("float64")
            amount = g["amount"].astype("float64")
            avg_long = float(last_px.rolling(max(1, sum_window_long), min_periods=1).mean().iloc[-1])
            std_long = float(
                last_px.rolling(max(2, sum_window_long), min_periods=2).std().fillna(0.0).iloc[-1]
            )
            avg_short = float(last_px.rolling(max(1, sum_window_short), min_periods=1).mean().iloc[-1])
            upper_band = avg_long + std_long
            lower_band = avg_long - std_long
            adv = float(amount.rolling(max(1, adv_window), min_periods=1).mean().iloc[-1])
            vol_ratio = float(volume.iloc[-1] / (adv + EPS))
            if upper_band < avg_short:
                return -1.0
            if avg_short < lower_band:
                return 1.0
            return 1.0 if vol_ratio >= 1.0 else -1.0

        return _group_scalar(frame, _calc)

