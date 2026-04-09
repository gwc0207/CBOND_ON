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

@FactorRegistry.register("alpha020_open_delay_range_v1")
class Alpha020OpenDelayRangeV1Factor(_AlphaBase):
    name = "alpha020_open_delay_range_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delay_window = int(ctx.params.get("delay_window", 1))
        frame = _prepare_panel(ctx, ["open", "ask_price1", "bid_price1", "high", "low", "last"])

        def _d1(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            high = g["high"].astype("float64")
            return float(open_.iloc[-1] - _delay_last(high, delay_window))

        def _d2(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            last_px = g["last"].astype("float64")
            return float(open_.iloc[-1] - _delay_last(last_px, delay_window))

        def _d3(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            low = g["low"].astype("float64")
            return float(open_.iloc[-1] - _delay_last(low, delay_window))

        d1 = _group_scalar(frame, _d1)
        d2 = _group_scalar(frame, _d2)
        d3 = _group_scalar(frame, _d3)
        return (-_cs_rank(d1)) * _cs_rank(d2) * _cs_rank(d3)


