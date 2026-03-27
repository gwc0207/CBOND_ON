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

@FactorRegistry.register("alpha016_cov_high_volume_v1")
class Alpha016CovHighVolumeV1Factor(_AlphaBase):
    name = "alpha016_cov_high_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        cov_window = int(ctx.params.get("cov_window", 5))
        frame = _prepare_panel(ctx, ["high", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            high_rank = g["high"].astype("float64").rank(pct=True, method="average")
            vol_rank = g["volume"].astype("float64").rank(pct=True, method="average")
            return _cov_last(high_rank, vol_rank, cov_window)

        raw = _group_scalar(frame, _calc)
        return -_cs_rank(raw)

