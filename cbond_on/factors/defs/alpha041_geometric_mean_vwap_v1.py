from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha041_geometric_mean_vwap_v1")
class Alpha041GeometricMeanVwapV1Factor(_AlphaBase):
    name = "alpha041_geometric_mean_vwap_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        frame = _prepare_panel(ctx, ["high", "low", "amount", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            high = g["high"].astype("float64").clip(lower=0.0)
            low = g["low"].astype("float64").clip(lower=0.0)
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            geo_mean = np.sqrt(high * low)
            vwap = amount / (volume + EPS)
            return float(geo_mean.iloc[-1] - vwap.iloc[-1])

        return _group_scalar(frame, _calc)

