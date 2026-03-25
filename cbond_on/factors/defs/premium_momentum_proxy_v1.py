from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._bond_stock_utils import build_bond_stock_latest_frame, to_dt_code_series


@FactorRegistry.register("premium_momentum_proxy_v1")
class PremiumMomentumProxyV1Factor(Factor):
    name = "premium_momentum_proxy_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        frame = build_bond_stock_latest_frame(
            ctx,
            bond_cols=["last", "pre_close"],
            stock_cols=["last", "pre_close"],
        )
        if frame.empty:
            out = pd.Series(dtype="float64")
            out.name = self.output_name(self.name)
            return out

        bond_last = pd.to_numeric(frame["last"], errors="coerce")
        bond_pre_close = pd.to_numeric(frame["pre_close"], errors="coerce")
        stock_last = pd.to_numeric(frame["stock_last"], errors="coerce")
        stock_pre_close = pd.to_numeric(frame["stock_pre_close"], errors="coerce")

        bond_strength = (bond_last - bond_pre_close) / (bond_pre_close + 1e-8)
        stock_strength = (stock_last - stock_pre_close) / (stock_pre_close + 1e-8)
        values = bond_strength - stock_strength
        return to_dt_code_series(frame, values, name=self.output_name(self.name))

