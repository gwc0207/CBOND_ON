"""Parity-scaled stock/bond tail-return gap, available at T1430."""

from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext


def _normalize_stock_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 6:
        return text
    if digits[0] in {"5", "6", "9"}:
        return f"{digits}.SH"
    if digits[0] in {"4", "8"}:
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def _tail_return(panel: pd.DataFrame | None, *, signal_day: date) -> pd.DataFrame:
    """T-day first-to-last `last` return over the causal 14:00--14:29 window."""
    columns = ["code", "dt", "tail_return"]
    if panel is None or panel.empty:
        return pd.DataFrame(columns=columns)
    frame = panel.reset_index().copy()
    if "trade_time" not in frame.columns or "last" not in frame.columns:
        return pd.DataFrame(columns=columns)
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    trade_time = frame["trade_time"]
    frame = frame.loc[
        trade_time.notna()
        & (trade_time.dt.date == signal_day)
        & (trade_time.dt.time >= time(14, 0))
        & (trade_time.dt.time <= time(14, 29)),
        ["dt", "code", "trade_time", "last"],
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["code"] = frame["code"].astype(str).str.strip().str.upper()
    frame["last"] = pd.to_numeric(frame["last"], errors="coerce")
    frame = frame.dropna(subset=["code", "trade_time", "last"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.sort_values(["code", "trade_time"], kind="mergesort")
    result = frame.groupby("code", sort=False).agg(
        dt=("dt", "last"),
        start_last=("last", "first"),
        end_last=("last", "last"),
    ).reset_index()
    result["tail_return"] = result["end_last"] / result["start_last"] - 1.0
    result = result.loc[
        np.isfinite(pd.to_numeric(result["tail_return"], errors="coerce")),
        columns,
    ]
    return result


@FactorRegistry.register("parity_adjusted_stock_lag_v1")
class ParityAdjustedStockLagV1Factor(Factor):
    """Use T-1 parity and pool membership to scale the stock/cbond tail gap."""

    name = "parity_adjusted_stock_lag_v1"
    requires_stock_panel = True

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        params = dict(params or {})
        base_source = str(params.get("daily_source", "market_cbond.daily_base")).strip()
        return [
            DailyFactorRequirement(
                source=base_source,
                columns=(
                    "instrument_code",
                    "exchange_code",
                    "stock_code",
                    "conv_value",
                    "cb_close_price",
                ),
                # Context contains T and T-1; compute explicitly selects T-1.
                lookback_days=2,
            ),
            DailyFactorRequirement(
                source="o_0005",
                columns=("instrument_code", "exchange_code"),
                lookback_days=2,
            ),
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        signal_day = pd.Timestamp(ctx.panel.attrs.get("__build_day__")).date()
        base_source = str(ctx.params.get("daily_source", "market_cbond.daily_base")).strip()
        base = ctx.daily_data.get(base_source)
        pool = ctx.daily_data.get("o_0005")
        cbond_return = _tail_return(ctx.panel, signal_day=signal_day).rename(
            columns={
                "code": "bond_code",
                "dt": "factor_dt",
                "tail_return": "cbond_tail_return",
            }
        )
        stock_return = _tail_return(ctx.stock_panel, signal_day=signal_day).rename(
            columns={
                "code": "stock_code",
                "dt": "stock_dt",
                "tail_return": "stock_tail_return",
            }
        )
        if (
            base is None
            or base.empty
            or pool is None
            or pool.empty
            or cbond_return.empty
            or stock_return.empty
        ):
            return pd.Series(dtype="float64", name=self.output_name(self.name))

        prev = base.copy()
        prev["trade_date"] = pd.to_datetime(prev["trade_date"], errors="coerce").dt.date
        required = {"code", "exchange_code", "stock_code", "conv_value", "cb_close_price"}
        if not required.issubset(prev.columns):
            return pd.Series(dtype="float64", name=self.output_name(self.name))
        prev = prev.loc[prev["trade_date"] < signal_day].sort_values(
            ["code", "trade_date"], kind="mergesort"
        )
        prev = prev.drop_duplicates(subset=["code"], keep="last")
        prev["bond_code"] = (
            prev["code"].astype(str).str.strip().str.zfill(6)
            + "."
            + prev["exchange_code"].astype(str).str.strip().str.upper()
        )
        prev["stock_code"] = prev["stock_code"].map(_normalize_stock_code)
        prev["conv_value"] = pd.to_numeric(prev["conv_value"], errors="coerce")
        prev["cb_close_price"] = pd.to_numeric(prev["cb_close_price"], errors="coerce")
        prev["kappa"] = (prev["conv_value"] / prev["cb_close_price"]).clip(0.0, 2.0)
        prev = prev.replace({"stock_code": {"": pd.NA}}).dropna(
            subset=["bond_code", "stock_code", "kappa"]
        )

        eligible = pool.copy()
        eligible["trade_date"] = pd.to_datetime(eligible["trade_date"], errors="coerce").dt.date
        if not {"code", "exchange_code"}.issubset(eligible.columns):
            return pd.Series(dtype="float64", name=self.output_name(self.name))
        eligible = eligible.loc[eligible["trade_date"] < signal_day].sort_values(
            ["code", "trade_date"], kind="mergesort"
        )
        eligible = eligible.drop_duplicates(subset=["code"], keep="last")
        eligible_codes = set(
            (
                eligible["code"].astype(str).str.strip().str.zfill(6)
                + "."
                + eligible["exchange_code"].astype(str).str.strip().str.upper()
            ).tolist()
        )
        if not eligible_codes:
            return pd.Series(dtype="float64", name=self.output_name(self.name))

        merged = cbond_return.merge(
            prev[["bond_code", "stock_code", "kappa"]], on="bond_code", how="inner"
        )
        merged = merged.merge(stock_return, on="stock_code", how="inner")
        merged = merged.loc[merged["bond_code"].isin(eligible_codes)].copy()
        if merged.empty:
            return pd.Series(dtype="float64", name=self.output_name(self.name))
        merged["factor_value"] = (
            merged["kappa"] * merged["stock_tail_return"] - merged["cbond_tail_return"]
        )
        merged["factor_value"] = pd.to_numeric(merged["factor_value"], errors="coerce")
        merged = merged.dropna(subset=["factor_dt", "bond_code", "factor_value"])
        index = pd.MultiIndex.from_frame(
            merged[["factor_dt", "bond_code"]], names=["dt", "code"]
        )
        output = pd.Series(merged["factor_value"].to_numpy(), index=index, dtype="float64")
        output = output[~output.index.duplicated(keep="last")]
        output.name = self.output_name(self.name)
        return output
