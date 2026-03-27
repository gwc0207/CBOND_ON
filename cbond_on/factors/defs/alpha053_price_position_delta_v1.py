from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha053_price_position_delta_v1")
class Alpha053PricePositionDeltaV1Factor(_AlphaBase):
    name = "alpha053_price_position_delta_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delta_window = int(ctx.params.get("delta_window", 9))
        frame = _prepare_panel(ctx, ["last", "high", "low"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            high = g["high"].astype("float64")
            low = g["low"].astype("float64")
            numerator = (last_px - low) - (high - last_px)
            denominator = last_px - low
            position = numerator / (denominator + EPS)
            delta_val = position.diff(max(1, delta_window)).iloc[-1]
            if pd.isna(delta_val):
                return 0.0
            return float(-delta_val)

        return _group_scalar(frame, _calc)

