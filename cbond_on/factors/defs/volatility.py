from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, group_apply_scalar, slice_window


@FactorRegistry.register("volatility")
class VolatilityFactor(Factor):
    name = "volatility"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        price_col = str(ctx.params.get("price_col", "last"))
        use_log = bool(ctx.params.get("use_log_return", True))
        if price_col not in panel.columns:
            raise KeyError(f"volatility missing column: {price_col}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            if df.empty:
                return 0.0
            price = df[price_col].astype("float64")
            if use_log:
                ret = np.log(price.replace(0, np.nan)).diff()
            else:
                ret = price.pct_change()
            ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
            if ret.empty:
                return 0.0
            return float(ret.std())

        out = group_apply_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
