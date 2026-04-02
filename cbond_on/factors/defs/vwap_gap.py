from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("vwap_gap")
class VwapGapFactor(Factor):
    name = "vwap_gap"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        price_col = str(ctx.params.get("price_col", "last"))
        volume_col = str(ctx.params.get("volume_col", "volume"))
        amount_col = str(ctx.params.get("amount_col", "amount"))
        required = [price_col, volume_col, amount_col]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise KeyError(f"vwap_gap missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            if df.empty:
                return 0.0
            last_price = float(df[price_col].astype("float64").iloc[-1])
            amount = float(df[amount_col].astype("float64").sum())
            volume = float(df[volume_col].astype("float64").sum())
            if volume <= 0:
                return 0.0
            vwap = amount / volume
            if vwap <= 0:
                return 0.0
            return float((last_price - vwap) / vwap)

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out

