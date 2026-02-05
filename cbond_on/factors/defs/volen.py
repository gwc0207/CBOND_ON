from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext, ensure_panel_index


@FactorRegistry.register("volen")
class VolenFactor(Factor):
    name = "volen"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_panel_index(ctx.panel)
        levels = int(ctx.params.get("levels", 5))
        fast = int(ctx.params.get("fast", 60))
        slow = int(ctx.params.get("slow", 10))
        if fast <= 0 or slow <= 0:
            raise ValueError("fast/slow must be > 0")

        ask_cols = [f"ask_volume{i}" for i in range(1, levels + 1)]
        bid_cols = [f"bid_volume{i}" for i in range(1, levels + 1)]
        missing = [c for c in ask_cols + bid_cols if c not in panel.columns]
        if missing:
            raise KeyError(f"volen missing columns: {missing}")

        total_vol = panel[ask_cols].sum(axis=1) + panel[bid_cols].sum(axis=1)
        total_vol = total_vol.astype("float64")

        grouped = total_vol.groupby(level=["dt", "code"])
        fast_mean = grouped.rolling(window=fast, min_periods=fast).mean()
        slow_mean = grouped.rolling(window=slow, min_periods=slow).mean()
        fast_mean = fast_mean.reset_index(level=[0, 1], drop=True)
        slow_mean = slow_mean.reset_index(level=[0, 1], drop=True)

        fast_last = fast_mean.groupby(level=["dt", "code"]).tail(1)
        slow_last = slow_mean.groupby(level=["dt", "code"]).tail(1)

        fast_last = fast_last.droplevel("seq")
        slow_last = slow_last.droplevel("seq")

        ratio = fast_last.div(slow_last.replace(0, pd.NA))
        ratio = ratio.fillna(0.0)
        ratio.name = self.output_name(self.name)
        return ratio
