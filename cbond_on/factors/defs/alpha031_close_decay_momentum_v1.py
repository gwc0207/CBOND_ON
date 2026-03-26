from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import _AlphaBase, _corr_last, _cs_rank, _delta_last, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha031_close_decay_momentum_v1")
class Alpha031CloseDecayMomentumV1Factor(_AlphaBase):
    name = "alpha031_close_decay_momentum_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delta_window = int(ctx.params.get("delta_window", 10))
        decay_window = int(ctx.params.get("decay_window", 10))
        delta_short_window = int(ctx.params.get("delta_short_window", 3))
        corr_window = int(ctx.params.get("corr_window", 12))
        adv_window = int(ctx.params.get("adv_window", 10))
        frame = _prepare_panel(ctx, ["last", "amount", "low"])

        def _decay_term(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            delta_close = last_px.diff(max(1, delta_window))
            rank_delta = delta_close.rank(pct=True, method="average")
            tail = (-rank_delta).tail(max(1, decay_window)).fillna(0.0)
            weights = np.arange(1, len(tail) + 1, dtype="float64")
            if float(weights.sum()) <= 0:
                return 0.0
            return float(np.dot(tail.to_numpy(dtype="float64"), weights) / weights.sum())

        def _short_mom(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            return float(-_delta_last(last_px, delta_short_window))

        def _corr_term(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            low = g["low"].astype("float64")
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            return float(_corr_last(adv, low, corr_window))

        decay_term = _group_scalar(frame, _decay_term)
        short_mom = _group_scalar(frame, _short_mom)
        corr_term = _group_scalar(frame, _corr_term)
        corr_sign = np.sign(_cs_rank(corr_term) - 0.5)
        return _cs_rank(decay_term) + _cs_rank(short_mom) + corr_sign

