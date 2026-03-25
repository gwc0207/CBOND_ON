from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._bond_stock_utils import build_bond_stock_latest_frame, to_dt_code_series


@FactorRegistry.register("stock_bond_momentum_gap_v1")
class StockBondMomentumGapV1Factor(Factor):
    name = "stock_bond_momentum_gap_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        frame = build_bond_stock_latest_frame(
            ctx,
            bond_cols=["last", "open"],
            stock_cols=["last", "open"],
        )
        if frame.empty:
            out = pd.Series(dtype="float64")
            out.name = self.output_name(self.name)
            return out

        bond_last = pd.to_numeric(frame["last"], errors="coerce")
        bond_open = pd.to_numeric(frame["open"], errors="coerce")
        stock_last = pd.to_numeric(frame["stock_last"], errors="coerce")
        stock_open = pd.to_numeric(frame["stock_open"], errors="coerce")

        bond_ret = (bond_last - bond_open) / (bond_open + 1e-8)
        stock_ret = (stock_last - stock_open) / (stock_open + 1e-8)
        values = bond_ret - stock_ret
        return to_dt_code_series(frame, values, name=self.output_name(self.name))

