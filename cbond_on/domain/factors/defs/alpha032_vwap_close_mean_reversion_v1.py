from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _corr_last, _cs_rank, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha032_vwap_close_mean_reversion_v1")
class Alpha032VwapCloseMeanReversionV1Factor(_AlphaBase):
    name = "alpha032_vwap_close_mean_reversion_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        sum_window = int(ctx.params.get("sum_window", 7))
        corr_window = int(ctx.params.get("corr_window", 60))
        delay_window = int(ctx.params.get("delay_window", 5))
        corr_scale = float(ctx.params.get("corr_scale", 20.0))
        frame = _prepare_panel(ctx, ["last", "amount", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            avg_close = float(last_px.rolling(max(1, sum_window), min_periods=1).mean().iloc[-1])
            diff1 = avg_close - float(last_px.iloc[-1])
            vwap = amount / (volume + EPS)
            delayed_last = last_px.shift(max(1, delay_window))
            corr = _corr_last(vwap, delayed_last, corr_window)
            return float(diff1 + corr_scale * corr)

        raw = _group_scalar(frame, _calc)
        return _cs_rank(raw)



