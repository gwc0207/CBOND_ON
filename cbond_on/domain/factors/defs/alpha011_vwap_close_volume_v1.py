from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import (
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

@FactorRegistry.register("alpha011_vwap_close_volume_v1")
class Alpha011VwapCloseVolumeV1Factor(_AlphaBase):
    name = "alpha011_vwap_close_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_window = int(ctx.params.get("ts_window", 3))
        frame = _prepare_panel(ctx, ["amount", "volume", "last"])

        def _max_diff(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            vwap = amount / (volume + EPS)
            diff = vwap - last_px
            return float(diff.rolling(max(1, ts_window), min_periods=1).max().iloc[-1])

        def _min_diff(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            vwap = amount / (volume + EPS)
            diff = vwap - last_px
            return float(diff.rolling(max(1, ts_window), min_periods=1).min().iloc[-1])

        def _delta_vol(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64")
            return _delta_last(volume, ts_window)

        max_diff = _group_scalar(frame, _max_diff)
        min_diff = _group_scalar(frame, _min_diff)
        delta_vol = _group_scalar(frame, _delta_vol)
        return (_cs_rank(max_diff) + _cs_rank(min_diff)) * _cs_rank(delta_vol)


