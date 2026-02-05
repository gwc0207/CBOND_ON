from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, group_apply_scalar, slice_window


@FactorRegistry.register("mom_slope")
class MomentumSlopeFactor(Factor):
    name = "mom_slope"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        price_col = str(ctx.params.get("price_col", "last"))
        if price_col not in panel.columns:
            raise KeyError(f"mom_slope missing column: {price_col}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            if df.empty:
                return 0.0
            y = df[price_col].astype("float64").values
            if y.size < 2:
                return 0.0
            t0 = df["trade_time"].iloc[0]
            x = (df["trade_time"] - t0).dt.total_seconds().values
            if np.allclose(x, 0):
                return 0.0
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)

        out = group_apply_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
