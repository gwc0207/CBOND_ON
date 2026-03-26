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

@FactorRegistry.register("alpha015_high_volume_corr_v1")
class Alpha015HighVolumeCorrV1Factor(_AlphaBase):
    name = "alpha015_high_volume_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 3))
        sum_window = int(ctx.params.get("sum_window", 3))
        frame = _prepare_panel(ctx, ["high", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            high_rank = g["high"].astype("float64").rank(pct=True, method="average")
            vol_rank = g["volume"].astype("float64").rank(pct=True, method="average")
            corr_series = high_rank.rolling(max(2, corr_window), min_periods=2).corr(vol_rank)
            corr_series = corr_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            ranked_corr = corr_series.rank(pct=True, method="average")
            return float(ranked_corr.tail(max(1, sum_window)).sum())

        raw = _group_scalar(frame, _calc)
        return -raw
