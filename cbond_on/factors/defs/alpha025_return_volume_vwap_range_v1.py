from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import EPS, _AlphaBase, _cs_rank, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha025_return_volume_vwap_range_v1")
class Alpha025ReturnVolumeVwapRangeV1Factor(_AlphaBase):
    name = "alpha025_return_volume_vwap_range_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 10))
        frame = _prepare_panel(ctx, ["last", "pre_close", "amount", "volume", "high"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["pre_close"].astype("float64")
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            high = g["high"].astype("float64")
            returns = float((last_px.iloc[-1] - pre_close.iloc[-1]) / (pre_close.iloc[-1] + EPS))
            adv = float(amount.rolling(max(1, adv_window), min_periods=1).mean().iloc[-1])
            vwap = float(amount.iloc[-1] / (volume.iloc[-1] + EPS))
            price_range = float(high.iloc[-1] - last_px.iloc[-1])
            return float(((-returns) * adv) * vwap * price_range)

        raw = _group_scalar(frame, _calc)
        return _cs_rank(raw)

