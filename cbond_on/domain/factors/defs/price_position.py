from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("price_position")
class PricePositionFactor(Factor):
    name = "price_position"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        price_col = str(ctx.params.get("price_col", "last"))
        if price_col not in panel.columns:
            raise KeyError(f"price_position missing column: {price_col}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            if df.empty:
                return 0.0
            price = df[price_col].astype("float64")
            lo = price.min()
            hi = price.max()
            if hi <= lo:
                return 0.0
            return float((price.iloc[-1] - lo) / (hi - lo))

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out

