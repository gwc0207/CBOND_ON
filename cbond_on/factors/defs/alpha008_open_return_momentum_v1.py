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

@FactorRegistry.register("alpha008_open_return_momentum_v1")
class Alpha008OpenReturnMomentumV1Factor(_AlphaBase):
    name = "alpha008_open_return_momentum_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        sum_window = int(ctx.params.get("sum_window", 5))
        delay_window = int(ctx.params.get("delay_window", 10))
        frame = _prepare_panel(ctx, ["open", "ask_price1", "bid_price1", "last"])

        def _calc(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            last_px = g["last"].astype("float64")
            ret = (last_px - open_) / (open_ + EPS)
            sum_open = open_.rolling(max(1, sum_window), min_periods=1).sum()
            sum_ret = ret.rolling(max(1, sum_window), min_periods=1).sum()
            prod = sum_open * sum_ret
            delayed = prod.shift(max(1, delay_window))
            delay_val = float(delayed.iloc[-1]) if pd.notna(delayed.iloc[-1]) else float(prod.iloc[0])
            return float(prod.iloc[-1] - delay_val)

        raw = _group_scalar(frame, _calc)
        return -_cs_rank(raw)

