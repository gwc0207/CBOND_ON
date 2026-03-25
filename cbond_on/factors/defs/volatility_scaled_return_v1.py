from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, group_apply_scalar


@FactorRegistry.register("volatility_scaled_return_v1")
class VolatilityScaledReturnV1Factor(Factor):
    name = "volatility_scaled_return_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required = ["last", "open", "high", "low", "pre_close"]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise KeyError(f"volatility_scaled_return_v1 missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            row = df.sort_values("trade_time").iloc[-1]
            last = float(row["last"])
            open_px = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            pre_close = float(row["pre_close"])
            intraday_range = (high - low) / (pre_close + 1e-8)
            if intraday_range <= 0:
                return 0.0
            ret = (last - open_px) / (open_px + 1e-8)
            return float(ret / (intraday_range + 1e-8))

        out = group_apply_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out


