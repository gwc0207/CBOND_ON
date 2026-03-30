from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import (
    EPS,
    _AlphaBase,
    _corr_last,
    _cov_last,
    _cs_rank,
    _delay_last,
    _delta_last,
    _group_scalar,
    _open_like,
    _prepare_panel,
    _ts_rank_last,
)

@FactorRegistry.register("alpha005_vwap_gap_v1")
class Alpha005VwapGapV1Factor(_AlphaBase):
    name = "alpha005_vwap_gap_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        vwap_window = int(ctx.params.get("vwap_window", 10))
        frame = _prepare_panel(ctx, ["amount", "volume", "open", "ask_price1", "bid_price1", "last"])

        def _gap1(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            open_ = _open_like(g)
            vwap = amount / (volume + EPS)
            avg_vwap = vwap.rolling(max(1, vwap_window), min_periods=1).mean()
            return float(open_.iloc[-1] - avg_vwap.iloc[-1])

        def _gap2(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            vwap = amount / (volume + EPS)
            return float(last_px.iloc[-1] - vwap.iloc[-1])

        gap1 = _group_scalar(
            frame,
            _gap1,
            kernel_name="alpha005_vwap_gap_v1_gap1",
            kernel_params={"vwap_window": vwap_window},
        )
        gap2 = _group_scalar(
            frame,
            _gap2,
            kernel_name="alpha005_vwap_gap_v1_gap2",
            kernel_params={},
        )
        return _cs_rank(gap1) * (-_cs_rank(gap2).abs())

