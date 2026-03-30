from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, ensure_trade_time, group_apply_scalar


@FactorRegistry.register("aacb")
class AacbFactor(Factor):
    name = "aacb"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        levels = int(ctx.params.get("levels", 3))
        ask_cols = [f"ask_price{i}" for i in range(1, levels + 1)]
        bid_cols = [f"bid_price{i}" for i in range(1, levels + 1)]
        missing = [c for c in ask_cols + bid_cols if c not in panel.columns]
        if missing:
            raise KeyError(f"aacb missing columns: {missing}")

        def _calc(g: pd.DataFrame) -> float:
            ask_sum = None
            bid_sum = None
            for col in ask_cols:
                s = g[col].astype("float64")
                ask_sum = s if ask_sum is None else (ask_sum + s)
            for col in bid_cols:
                s = g[col].astype("float64")
                bid_sum = s if bid_sum is None else (bid_sum + s)
            ask_avg = ask_sum / float(max(1, len(ask_cols)))
            bid_avg = bid_sum / float(max(1, len(bid_cols)))
            mid = (g["ask_price1"].astype("float64") + g["bid_price1"].astype("float64")) / 2.0
            spread = ask_avg - bid_avg
            base = mid.where(abs(mid) > EPS, float("nan"))
            value = (spread / base).fillna(0.0)
            return float(value.mean())

        out = group_apply_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out
