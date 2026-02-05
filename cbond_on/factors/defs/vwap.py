from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, group_apply_scalar, slice_window


@FactorRegistry.register("vwap")
class VwapFactor(Factor):
    name = "vwap"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        volume_col = str(ctx.params.get("volume_col", "volume"))
        amount_col = str(ctx.params.get("amount_col", "amount"))
        if volume_col not in panel.columns or amount_col not in panel.columns:
            missing = [c for c in [volume_col, amount_col] if c not in panel.columns]
            raise KeyError(f"vwap missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            if df.empty:
                return 0.0
            vol = df[volume_col].astype("float64").sum()
            if vol <= 0:
                return 0.0
            amt = df[amount_col].astype("float64").sum()
            return float(amt / vol)

        out = group_apply_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
