from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import EPS, _AlphaBase, _cs_rank, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha027_volume_vwap_corr_signal_v1")
class Alpha027VolumeVwapCorrSignalV1Factor(_AlphaBase):
    name = "alpha027_volume_vwap_corr_signal_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 6))
        sum_window = int(ctx.params.get("sum_window", 2))
        frame = _prepare_panel(ctx, ["amount", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            vwap = amount / (volume + EPS)
            r1 = volume.rank(pct=True, method="average")
            r2 = vwap.rank(pct=True, method="average")
            corr = r1.rolling(max(2, corr_window), min_periods=2).corr(r2)
            avg_corr = float(corr.rolling(max(1, sum_window), min_periods=1).mean().iloc[-1])
            if pd.isna(avg_corr):
                return 0.0
            return avg_corr

        avg_corr = _group_scalar(frame, _calc)
        cond = _cs_rank(avg_corr) > 0.5
        return pd.Series(np.where(cond, -1.0, 1.0), index=avg_corr.index, dtype="float64")

