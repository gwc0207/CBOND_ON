from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, _group_scalar


@FactorRegistry.register("price_level_position_v1")
class PriceLevelPositionV1Factor(Factor):
    name = "price_level_position_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required = ["last", "high", "low"]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise KeyError(f"price_level_position_v1 missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            row = df.sort_values("trade_time").iloc[-1]
            last = float(row["last"])
            high = float(row["high"])
            low = float(row["low"])
            spread = high - low
            if spread <= 0:
                return 0.0
            return float((last - low) / (spread + 1e-8))

        out = _group_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out


