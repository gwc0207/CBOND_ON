from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import (
    _AlphaBase,
    _cs_rank,
    _group_scalar,
    _prepare_panel,
)


@FactorRegistry.register("alpha022_high_volume_corr_change_v1")
class Alpha022HighVolumeCorrChangeV1Factor(_AlphaBase):
    name = "alpha022_high_volume_corr_change_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 5))
        delta_window = int(ctx.params.get("delta_window", 5))
        stddev_window = int(ctx.params.get("stddev_window", 10))
        frame = _prepare_panel(ctx, ["high", "volume", "last"])

        def _delta_corr(g: pd.DataFrame) -> float:
            high = g["high"].astype("float64")
            volume = g["volume"].astype("float64")
            corr = high.rolling(max(2, corr_window), min_periods=2).corr(volume)
            val = corr.iloc[-1] - corr.shift(max(1, delta_window)).iloc[-1]
            if pd.isna(val):
                return 0.0
            return float(val)

        def _std_close(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            val = last_px.rolling(max(2, stddev_window), min_periods=2).std().fillna(0.0).iloc[-1]
            return float(val)

        delta_corr = _group_scalar(frame, _delta_corr)
        std_close = _group_scalar(frame, _std_close)
        return -(delta_corr * _cs_rank(std_close))


