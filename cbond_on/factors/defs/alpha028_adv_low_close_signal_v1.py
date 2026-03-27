from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import _AlphaBase, _corr_last, _cs_rank, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha028_adv_low_close_signal_v1")
class Alpha028AdvLowCloseSignalV1Factor(_AlphaBase):
    name = "alpha028_adv_low_close_signal_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 10))
        corr_window = int(ctx.params.get("corr_window", 5))
        frame = _prepare_panel(ctx, ["amount", "high", "low", "last"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            high = g["high"].astype("float64")
            low = g["low"].astype("float64")
            last_px = g["last"].astype("float64")
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            corr = _corr_last(adv, low, corr_window)
            mid_price = float(((high + low) / 2.0).iloc[-1])
            return float((corr + mid_price) - last_px.iloc[-1])

        raw = _group_scalar(frame, _calc)
        return _cs_rank(raw)


