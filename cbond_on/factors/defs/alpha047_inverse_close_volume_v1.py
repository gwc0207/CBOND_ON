from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha047_inverse_close_volume_v1")
class Alpha047InverseCloseVolumeV1Factor(_AlphaBase):
    name = "alpha047_inverse_close_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 10))
        sum_window = int(ctx.params.get("sum_window", 5))
        delay_window = int(ctx.params.get("delay_window", 5))
        power = int(ctx.params.get("power", 5))
        frame = _prepare_panel(ctx, ["last", "amount", "volume", "high"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            high = g["high"].astype("float64")

            inv_close = 1.0 / (last_px + EPS)
            rank_inv = inv_close.rank(pct=True, method="average")
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            vol_ratio = (rank_inv * volume) / (adv + EPS)

            rank_high_diff = (high - last_px).rank(pct=True, method="average")
            avg_high = high.rolling(max(1, sum_window), min_periods=1).mean()
            high_factor = (high * rank_high_diff) / (avg_high + EPS)

            vwap = amount / (volume + EPS)
            delay_vwap = vwap.shift(max(1, delay_window))
            rank_diff = (vwap - delay_vwap).rank(pct=True, method="average")
            alpha = vol_ratio * high_factor - rank_diff
            val = alpha.iloc[-1]
            if pd.isna(val):
                return 0.0
            return float(val)

        return _group_scalar(frame, _calc)

