from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import _group_scalar, _open_like, _prepare_panel


@FactorRegistry.register("volatility_scaled_return_v1")
class VolatilityScaledReturnV1Factor(Factor):
    name = "volatility_scaled_return_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        frame = _prepare_panel(
            ctx,
            ["last", "high", "low", "prev_bar_close", "open", "ask_price1", "bid_price1"],
        )

        def _calc(g: pd.DataFrame) -> float:
            row = g.iloc[-1]
            last = float(row["last"])
            open_px = float(_open_like(g).iloc[-1])
            high = float(row["high"])
            low = float(row["low"])
            pre_close = float(row["prev_bar_close"])
            intraday_range = (high - low) / (pre_close + 1e-8)
            if intraday_range <= 0:
                return 0.0
            ret = (last - open_px) / (open_px + 1e-8)
            return float(ret / (intraday_range + 1e-8))

        out = _group_scalar(frame, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out




