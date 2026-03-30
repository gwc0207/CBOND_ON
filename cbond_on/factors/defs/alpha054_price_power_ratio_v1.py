from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _open_like, _prepare_panel


@FactorRegistry.register("alpha054_price_power_ratio_v1")
class Alpha054PricePowerRatioV1Factor(_AlphaBase):
    name = "alpha054_price_power_ratio_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        power = int(ctx.params.get("power", 5))
        frame = _prepare_panel(ctx, ["low", "high", "last", "open", "ask_price1", "bid_price1"])

        def _calc(g: pd.DataFrame) -> float:
            low = g["low"].astype("float64")
            high = g["high"].astype("float64")
            last_px = g["last"].astype("float64")
            mid = _open_like(g).astype("float64")

            numerator = -(low - last_px) * mid.pow(power)
            denominator = (low - high) * last_px.pow(power)
            alpha = numerator / (denominator + EPS)
            val = alpha.iloc[-1]
            if pd.isna(val):
                return 0.0
            return float(val)

        return _group_scalar(frame, _calc)
