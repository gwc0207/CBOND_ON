from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import _AlphaBase, _group_scalar, _prepare_panel


def _rolling_ts_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(max(1, window), min_periods=1).apply(
        lambda x: x.rank(pct=True, method="average").iloc[-1],
        raw=False,
    )


@FactorRegistry.register("alpha026_volume_high_rank_corr_v1")
class Alpha026VolumeHighRankCorrV1Factor(_AlphaBase):
    name = "alpha026_volume_high_rank_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_rank_window = int(ctx.params.get("ts_rank_window", 5))
        corr_window = int(ctx.params.get("corr_window", 5))
        ts_max_window = int(ctx.params.get("ts_max_window", 3))
        frame = _prepare_panel(ctx, ["volume", "high"])

        def _calc(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64")
            high = g["high"].astype("float64")
            ts_rank_vol = _rolling_ts_rank(volume, ts_rank_window)
            ts_rank_high = _rolling_ts_rank(high, ts_rank_window)
            corr = ts_rank_vol.rolling(max(2, corr_window), min_periods=2).corr(ts_rank_high)
            ts_max = corr.rolling(max(1, ts_max_window), min_periods=1).max().iloc[-1]
            if pd.isna(ts_max):
                return 0.0
            return float(-ts_max)

        return _group_scalar(frame, _calc)

