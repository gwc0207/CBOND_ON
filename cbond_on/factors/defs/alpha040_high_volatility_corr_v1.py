from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import _AlphaBase, _corr_last, _cs_rank, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha040_high_volatility_corr_v1")
class Alpha040HighVolatilityCorrV1Factor(_AlphaBase):
    name = "alpha040_high_volatility_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        stddev_window = int(ctx.params.get("stddev_window", 10))
        corr_window = int(ctx.params.get("corr_window", 10))
        frame = _prepare_panel(ctx, ["high", "volume"])

        def _std_high(g: pd.DataFrame) -> float:
            high = g["high"].astype("float64")
            return float(high.rolling(max(2, stddev_window), min_periods=2).std().fillna(0.0).iloc[-1])

        def _corr_hv(g: pd.DataFrame) -> float:
            high = g["high"].astype("float64")
            volume = g["volume"].astype("float64")
            return float(_corr_last(high, volume, corr_window))

        std_high = _group_scalar(frame, _std_high)
        corr_hv = _group_scalar(frame, _corr_hv)
        return (-_cs_rank(std_high)) * corr_hv

