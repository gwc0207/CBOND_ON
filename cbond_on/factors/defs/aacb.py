from __future__ import annotations

import pandas as pd

from cbond.core.registry import FactorRegistry
from cbond.factors.base import Factor, FactorComputeContext, ensure_panel_index


@FactorRegistry.register("aacb")
class AacbFactor(Factor):
    name = "aacb"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_panel_index(ctx.panel)
        levels = int(ctx.params.get("levels", 3))
        ask_cols = [f"ask_price{i}" for i in range(1, levels + 1)]
        bid_cols = [f"bid_price{i}" for i in range(1, levels + 1)]
        missing = [c for c in ask_cols + bid_cols if c not in panel.columns]
        if missing:
            raise KeyError(f"aacb missing columns: {missing}")

        ask_avg = panel[ask_cols].mean(axis=1)
        bid_avg = panel[bid_cols].mean(axis=1)
        mid = (panel["ask_price1"] + panel["bid_price1"]) / 2.0
        spread = ask_avg - bid_avg
        base = mid.replace(0, pd.NA)
        value = spread.div(base)
        value = value.fillna(0.0)

        out = value.groupby(level=["dt", "code"]).mean()
        out.name = self.output_name(self.name)
        return out
