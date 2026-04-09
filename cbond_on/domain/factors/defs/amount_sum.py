from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("amount_sum")
class AmountSumFactor(Factor):
    name = "amount_sum"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        amount_col = str(ctx.params.get("amount_col", "amount"))
        if amount_col not in panel.columns:
            raise KeyError(f"amount_sum missing column: {amount_col}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            if df.empty:
                return 0.0
            return float(df[amount_col].astype("float64").sum())

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out

