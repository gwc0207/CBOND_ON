from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("midprice_move")
class MidpriceMoveFactor(Factor):
    name = "midprice_move"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        bid_col = str(ctx.params.get("bid_col", "bid_price1"))
        ask_col = str(ctx.params.get("ask_col", "ask_price1"))
        if bid_col not in panel.columns or ask_col not in panel.columns:
            missing = [c for c in [bid_col, ask_col] if c not in panel.columns]
            raise KeyError(f"midprice_move missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            if df.empty:
                return 0.0
            mid = (df[bid_col].astype("float64") + df[ask_col].astype("float64")) / 2.0
            first = mid.iloc[0]
            if first == 0:
                return 0.0
            return float((mid.iloc[-1] - first) / first)

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
