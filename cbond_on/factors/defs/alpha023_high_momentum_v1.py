from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import _AlphaBase, _delta_last, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha023_high_momentum_v1")
class Alpha023HighMomentumV1Factor(_AlphaBase):
    name = "alpha023_high_momentum_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        sum_window = int(ctx.params.get("sum_window", 10))
        delta_window = int(ctx.params.get("delta_window", 2))
        frame = _prepare_panel(ctx, ["high"])

        def _calc(g: pd.DataFrame) -> float:
            high = g["high"].astype("float64")
            avg_high = float(high.rolling(max(1, sum_window), min_periods=1).mean().iloc[-1])
            if avg_high < float(high.iloc[-1]):
                return float(-_delta_last(high, delta_window))
            return 0.0

        return _group_scalar(frame, _calc)

