from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, open_like_series


@FactorRegistry.register("intraday_momentum_v1")
class IntradayMomentumV1Factor(Factor):
    name = "intraday_momentum_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        if "last" not in panel.columns:
            raise KeyError("intraday_momentum_v1 missing columns: ['last']")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            row = df.iloc[-1]
            last = float(row["last"])
            open_px = float(open_like_series(df).iloc[-1])
            return float((last - open_px) / (open_px + 1e-8))

        out = _group_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out



