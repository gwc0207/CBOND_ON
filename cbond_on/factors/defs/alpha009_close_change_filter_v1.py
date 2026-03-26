from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import (
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

@FactorRegistry.register("alpha009_close_change_filter_v1")
class Alpha009CloseChangeFilterV1Factor(_AlphaBase):
    name = "alpha009_close_change_filter_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_window = int(ctx.params.get("ts_window", 5))
        frame = _prepare_panel(ctx, ["last"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            delta_last = last_px.diff(1)
            ts_min = delta_last.rolling(max(1, ts_window), min_periods=1).min().iloc[-1]
            ts_max = delta_last.rolling(max(1, ts_window), min_periods=1).max().iloc[-1]
            d = float(delta_last.iloc[-1]) if pd.notna(delta_last.iloc[-1]) else 0.0
            if float(ts_min) > 0.0:
                return d
            if float(ts_max) < 0.0:
                return d
            return -d

        return _group_scalar(frame, _calc)
