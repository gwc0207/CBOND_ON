from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, group_apply_scalar


@FactorRegistry.register("trade_intensity_v1")
class TradeIntensityV1Factor(Factor):
    name = "trade_intensity_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required = ["amount", "num_trades", "pre_close"]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise KeyError(f"trade_intensity_v1 missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            row = df.sort_values("trade_time").iloc[-1]
            amount = float(row["amount"])
            num_trades = float(row["num_trades"])
            pre_close = float(row["pre_close"])
            avg_trade_size = amount / (num_trades + 1e-8)
            return float(avg_trade_size / (pre_close + 1e-8))

        out = group_apply_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out


