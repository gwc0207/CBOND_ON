from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import (
    EPS,
    _AlphaBase,
    _delay_last,
    _group_scalar,
    _prepare_panel,
    _ts_rank_last,
)


@FactorRegistry.register("alpha029_complex_rank_signal_v1")
class Alpha029ComplexRankSignalV1Factor(_AlphaBase):
    name = "alpha029_complex_rank_signal_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_min_window = int(ctx.params.get("ts_min_window", 2))
        ts_rank_window = int(ctx.params.get("ts_rank_window", 5))
        delay_window = int(ctx.params.get("delay_window", 3))
        min_window = int(ctx.params.get("min_window", 5))
        frame = _prepare_panel(ctx, ["last", "pre_close"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["pre_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            delta_close = last_px.diff(1)
            rank_delta = (-delta_close).rank(pct=True, method="average")
            ts_min_rank = rank_delta.rolling(max(1, ts_min_window), min_periods=1).min()
            sum_min = ts_min_rank
            log_sum = np.log(sum_min.abs() + EPS)
            scaled = (log_sum - log_sum.mean()) / (log_sum.abs().sum() + EPS)
            rank_scaled = scaled.rank(pct=True, method="average")
            rank_prod = rank_scaled
            min_rank = float(rank_prod.rolling(max(1, min_window), min_periods=1).min().iloc[-1])
            delay_ret = (-returns).shift(max(1, delay_window))
            ts_rank_ret = _ts_rank_last(delay_ret, ts_rank_window)
            return float(min_rank + ts_rank_ret)

        return _group_scalar(frame, _calc)

