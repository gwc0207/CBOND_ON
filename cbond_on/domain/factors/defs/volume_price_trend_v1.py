from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import _group_scalar, _prepare_panel


@FactorRegistry.register("volume_price_trend_v1")
class VolumePriceTrendV1Factor(Factor):
    name = "volume_price_trend_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        frame = _prepare_panel(ctx, ["last", "prev_bar_close", "amount"])

        def _calc(g: pd.DataFrame) -> float:
            row = g.iloc[-1]
            last = float(row["last"])
            pre_close = float(row["prev_bar_close"])
            amount = float(row["amount"])
            ret = (last - pre_close) / (pre_close + 1e-8)
            return float(ret * amount)

        out = _group_scalar(frame, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out




