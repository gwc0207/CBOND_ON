from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import _AlphaBase, _corr_last, _cs_rank, _group_scalar, _open_like, _prepare_panel


@FactorRegistry.register("alpha037_open_close_correlation_v1")
class Alpha037OpenCloseCorrelationV1Factor(_AlphaBase):
    name = "alpha037_open_close_correlation_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 30))
        delay_window = int(ctx.params.get("delay_window", 1))
        frame = _prepare_panel(ctx, ["last", "open", "ask_price1", "bid_price1"])

        def _corr_term(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            delay_diff = (open_ - last_px).shift(max(1, delay_window))
            return float(_corr_last(delay_diff, last_px, corr_window))

        def _diff_term(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            return float((open_ - last_px).iloc[-1])

        corr_term = _group_scalar(frame, _corr_term)
        diff_term = _group_scalar(frame, _diff_term)
        return _cs_rank(corr_term) + _cs_rank(diff_term)



