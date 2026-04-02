from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window, first_last_price


@FactorRegistry.register("ret_window")
class ReturnWindowFactor(Factor):
    name = "ret_window"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        price_col = str(ctx.params.get("price_col", "last"))
        if price_col not in panel.columns:
            raise KeyError(f"ret_window missing column: {price_col}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            first, last = first_last_price(df, price_col)
            if first is None or first == 0:
                return 0.0
            return (last - first) / first

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
