from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, group_apply_scalar, slice_window


@FactorRegistry.register("volume_imbalance")
class VolumeImbalanceFactor(Factor):
    name = "volume_imbalance"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        levels = int(ctx.params.get("levels", 3))
        bid_cols = [f"bid_volume{i}" for i in range(1, levels + 1)]
        ask_cols = [f"ask_volume{i}" for i in range(1, levels + 1)]
        missing = [c for c in bid_cols + ask_cols if c not in panel.columns]
        if missing:
            raise KeyError(f"volume_imbalance missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            if df.empty:
                return 0.0
            bid = df[bid_cols].sum(axis=1).astype("float64").sum()
            ask = df[ask_cols].sum(axis=1).astype("float64").sum()
            denom = bid + ask
            if denom <= 0:
                return 0.0
            return float((bid - ask) / denom)

        out = group_apply_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
