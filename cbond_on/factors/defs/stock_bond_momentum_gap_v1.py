from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._bond_stock_utils import build_bond_stock_latest_frame, to_dt_code_series


def _open_like_from_cols(df: pd.DataFrame, *, open_col: str, ask_col: str, bid_col: str) -> pd.Series:
    mid: pd.Series | None = None
    if ask_col in df.columns and bid_col in df.columns:
        ask = pd.to_numeric(df[ask_col], errors="coerce")
        bid = pd.to_numeric(df[bid_col], errors="coerce")
        mid = (ask + bid) / 2.0
    open_px: pd.Series | None = None
    if open_col in df.columns:
        open_px = pd.to_numeric(df[open_col], errors="coerce")

    if mid is not None and open_px is not None:
        return mid.where(mid.notna(), open_px)
    if mid is not None:
        return mid
    if open_px is not None:
        return open_px
    return pd.Series(float("nan"), index=df.index, dtype="float64")


@FactorRegistry.register("stock_bond_momentum_gap_v1")
class StockBondMomentumGapV1Factor(Factor):
    name = "stock_bond_momentum_gap_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        frame = build_bond_stock_latest_frame(
            ctx,
            bond_cols=["last", "open", "ask_price1", "bid_price1"],
            stock_cols=["last", "open", "ask_price1", "bid_price1"],
        )
        if frame.empty:
            out = pd.Series(dtype="float64")
            out.name = self.output_name(self.name)
            return out

        bond_last = pd.to_numeric(frame["last"], errors="coerce")
        bond_open = _open_like_from_cols(
            frame,
            open_col="open",
            ask_col="ask_price1",
            bid_col="bid_price1",
        )
        stock_last = pd.to_numeric(frame["stock_last"], errors="coerce")
        stock_open = _open_like_from_cols(
            frame,
            open_col="stock_open",
            ask_col="stock_ask_price1",
            bid_col="stock_bid_price1",
        )

        bond_ret = (bond_last - bond_open) / (bond_open + 1e-8)
        stock_ret = (stock_last - stock_open) / (stock_open + 1e-8)
        values = bond_ret - stock_ret
        return to_dt_code_series(frame, values, name=self.output_name(self.name))
