from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _cs_rank, _group_scalar, _open_like, _prepare_panel


@FactorRegistry.register("alpha033_open_close_ratio_v1")
class Alpha033OpenCloseRatioV1Factor(_AlphaBase):
    name = "alpha033_open_close_ratio_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        frame = _prepare_panel(ctx, ["last", "open", "ask_price1", "bid_price1"])

        def _calc(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            last_px = g["last"].astype("float64")
            ratio = 1.0 - float(open_.iloc[-1] / (last_px.iloc[-1] + EPS))
            return float(-ratio)

        raw = _group_scalar(frame, _calc)
        return _cs_rank(raw)



