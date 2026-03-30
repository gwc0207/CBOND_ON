from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import _group_scalar, _prepare_panel


@FactorRegistry.register("trade_intensity_v1")
class TradeIntensityV1Factor(Factor):
    name = "trade_intensity_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        frame = _prepare_panel(ctx, ["amount", "num_trades", "prev_bar_close"])

        def _calc(g: pd.DataFrame) -> float:
            row = g.iloc[-1]
            amount = float(row["amount"])
            num_trades = float(row["num_trades"])
            pre_close = float(row["prev_bar_close"])
            avg_trade_size = amount / (num_trades + 1e-8)
            return float(avg_trade_size / (pre_close + 1e-8))

        out = _group_scalar(frame, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out



