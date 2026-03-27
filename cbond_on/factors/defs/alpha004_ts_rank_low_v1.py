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

@FactorRegistry.register("alpha004_ts_rank_low_v1")
class Alpha004TsRankLowV1Factor(_AlphaBase):
    name = "alpha004_ts_rank_low_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_rank_window = int(ctx.params.get("ts_rank_window", 9))
        frame = _prepare_panel(ctx, ["low"])

        def _calc(g: pd.DataFrame) -> float:
            low_rank = g["low"].astype("float64").rank(pct=True, method="average")
            return -_ts_rank_last(low_rank, ts_rank_window)

        return _group_scalar(frame, _calc)

