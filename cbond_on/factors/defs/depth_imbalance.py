from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, _group_scalar


@FactorRegistry.register("depth_imbalance")
class DepthImbalanceFactor(Factor):
    name = "depth_imbalance"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        levels = int(ctx.params.get("levels", 3))
        bid_cols = [f"bid_volume{i}" for i in range(1, levels + 1)]
        ask_cols = [f"ask_volume{i}" for i in range(1, levels + 1)]
        missing = [c for c in bid_cols + ask_cols if c not in panel.columns]
        if missing:
            raise KeyError(f"depth_imbalance missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            bid = df[bid_cols].iloc[-1].sum()
            ask = df[ask_cols].iloc[-1].sum()
            denom = bid + ask
            if denom <= 0:
                return 0.0
            return float((bid - ask) / denom)

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
