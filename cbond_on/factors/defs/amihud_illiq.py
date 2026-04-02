from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, first_last_price, _group_scalar, slice_window


@FactorRegistry.register("amihud_illiq")
class AmihudIlliqFactor(Factor):
    name = "amihud_illiq"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        price_col = str(ctx.params.get("price_col", "last"))
        amount_col = str(ctx.params.get("amount_col", "amount"))
        if price_col not in panel.columns:
            raise KeyError(f"amihud_illiq missing column: {price_col}")
        if amount_col not in panel.columns:
            raise KeyError(f"amihud_illiq missing column: {amount_col}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            if df.empty:
                return 0.0
            first, last = first_last_price(df, price_col)
            if first is None or first <= 0:
                return 0.0
            amount = float(df[amount_col].astype("float64").sum())
            if amount <= 0:
                return 0.0
            ret_abs = abs((last - first) / first)
            return float(ret_abs / amount)

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out

