from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _cs_rank, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha042_vwap_close_rank_ratio_v1")
class Alpha042VwapCloseRankRatioV1Factor(_AlphaBase):
    name = "alpha042_vwap_close_rank_ratio_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        frame = _prepare_panel(ctx, ["amount", "volume", "last"])

        def _diff(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            vwap = amount / (volume + EPS)
            return float(vwap.iloc[-1] - last_px.iloc[-1])

        def _sum_val(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            vwap = amount / (volume + EPS)
            return float(vwap.iloc[-1] + last_px.iloc[-1])

        diff = _group_scalar(frame, _diff)
        sum_val = _group_scalar(frame, _sum_val)
        return _cs_rank(diff) / (_cs_rank(sum_val) + EPS)

