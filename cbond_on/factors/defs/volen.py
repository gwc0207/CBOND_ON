from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, ensure_trade_time, group_apply_scalar


@FactorRegistry.register("volen")
class VolenFactor(Factor):
    name = "volen"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
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

        def _calc(g: pd.DataFrame) -> float:
            total = None
            for col in ask_cols + bid_cols:
                s = g[col].astype("float64")
                total = s if total is None else (total + s)
            fast_last = total.rolling(window=fast, min_periods=fast).mean().iloc[-1]
            slow_last = total.rolling(window=slow, min_periods=slow).mean().iloc[-1]
            if not pd.notna(fast_last) or not pd.notna(slow_last):
                return 0.0
            denom = float(slow_last)
            if abs(denom) <= EPS:
                return 0.0
            return float(fast_last) / denom

        ratio = group_apply_scalar(panel, _calc).fillna(0.0)
        ratio.name = self.output_name(self.name)
        return ratio
