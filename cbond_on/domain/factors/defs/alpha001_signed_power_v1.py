from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import (
    EPS,
    _AlphaBase,
    _corr_last,
    _cov_last,
    _cs_rank,
    _delay_last,
    _delta_last,
    _group_scalar,
    _open_like,
    _prepare_panel,
    _ts_rank_last,
)

@FactorRegistry.register("alpha001_signed_power_v1")
class Alpha001SignedPowerV1Factor(_AlphaBase):
    name = "alpha001_signed_power_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        stddev_window = int(ctx.params.get("stddev_window", 20))
        ts_max_window = int(ctx.params.get("ts_max_window", 5))
        frame = _prepare_panel(ctx, ["last", "prev_bar_close"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["prev_bar_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            std_ret = returns.rolling(max(2, stddev_window), min_periods=2).std().fillna(0.0)
            base = np.where(returns < 0.0, std_ret, last_px)
            sp = np.sign(base) * np.power(np.abs(base), 2.0)
            ts_max_sp = pd.Series(sp).rolling(max(1, ts_max_window), min_periods=1).max()
            return float(ts_max_sp.iloc[-1])

        raw = _group_scalar(
            frame,
            _calc,
            kernel_name="alpha001_signed_power_v1",
            kernel_params={
                "stddev_window": stddev_window,
                "ts_max_window": ts_max_window,
            },
        )
        return _cs_rank(raw) - 0.5



