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

@FactorRegistry.register("alpha019_close_momentum_sign_v1")
class Alpha019CloseMomentumSignV1Factor(_AlphaBase):
    name = "alpha019_close_momentum_sign_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delta_window = int(ctx.params.get("delta_window", 7))
        sum_window = int(ctx.params.get("sum_window", 250))
        frame = _prepare_panel(ctx, ["last", "prev_bar_close"])

        def _sign_term(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            delta_last = last_px.diff(max(1, delta_window))
            last_change = float(last_px.iloc[-1] - _delay_last(last_px, delta_window))
            delta_value = float(delta_last.iloc[-1]) if pd.notna(delta_last.iloc[-1]) else 0.0
            return float(np.sign(last_change + delta_value))

        def _sum_ret(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["prev_bar_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            return float(returns.rolling(max(1, sum_window), min_periods=1).sum().iloc[-1])

        sign_term = _group_scalar(frame, _sign_term)
        sum_ret = _group_scalar(frame, _sum_ret)
        return (-sign_term) * (1.0 + _cs_rank(1.0 + sum_ret))



