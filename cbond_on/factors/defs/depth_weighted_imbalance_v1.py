from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, _group_scalar


@FactorRegistry.register("depth_weighted_imbalance_v1")
class DepthWeightedImbalanceV1Factor(Factor):
    name = "depth_weighted_imbalance_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        weights_raw = ctx.params.get("weights", [5, 4, 3, 2, 1])
        if not isinstance(weights_raw, (list, tuple)) or not weights_raw:
            raise ValueError("depth_weighted_imbalance_v1 params.weights must be a non-empty list")
        weights = [float(x) for x in weights_raw]
        levels = len(weights)
        bid_cols = [f"bid_volume{i}" for i in range(1, levels + 1)]
        ask_cols = [f"ask_volume{i}" for i in range(1, levels + 1)]
        missing = [c for c in bid_cols + ask_cols if c not in panel.columns]
        if missing:
            raise KeyError(f"depth_weighted_imbalance_v1 missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            row = df.sort_values("trade_time").iloc[-1]
            bid = 0.0
            ask = 0.0
            for i, w in enumerate(weights, start=1):
                bid += float(row[f"bid_volume{i}"]) * w
                ask += float(row[f"ask_volume{i}"]) * w
            denom = bid + ask + 1e-8
            return float((bid - ask) / denom)

        out = _group_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out


