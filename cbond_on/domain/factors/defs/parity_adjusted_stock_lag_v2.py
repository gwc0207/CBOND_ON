"""Pool-free, T-1-mapped stock/bond tail-return gap for research only.

The factor is intentionally independent of any strategy pool or mask.  The
only daily inputs are taken from the latest daily_base observation strictly
before the signal day; intraday returns are taken from the T-day 14:00--14:29
interval that is visible by the T1430 cutoff.
"""

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


def _normalize_exchange(value: object) -> str:
    text = str(value or "").strip().upper()
    return {"XSHG": "SH", "SHSE": "SH", "XSHE": "SZ", "SZSE": "SZ"}.get(text, text)


def _signal_day(ctx: FactorComputeContext) -> date:
    raw = ctx.panel.attrs.get("__build_day__")
    if raw is None:
        raise ValueError("parity_adjusted_stock_lag_v2 missing panel __build_day__")
    parsed = pd.Timestamp(raw)
    if pd.isna(parsed):
        raise ValueError("parity_adjusted_stock_lag_v2 has invalid panel __build_day__")
    return parsed.date()


def _tail_return(
    panel: pd.DataFrame | None,
    *,
    signal_day: date,
    asset_name: str,
) -> pd.DataFrame:
    """Compute per-code first-to-last return over the causal 14:00--14:29 tail."""
    if panel is None or panel.empty:
        raise ValueError(f"parity_adjusted_stock_lag_v2 missing {asset_name} panel")
    if "trade_time" not in panel.columns or "last" not in panel.columns:
        missing = [col for col in ("trade_time", "last") if col not in panel.columns]
        raise KeyError(f"parity_adjusted_stock_lag_v2 {asset_name} panel missing columns: {missing}")

    frame = panel.reset_index().copy()
    required = {"dt", "code", "trade_time", "last"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"parity_adjusted_stock_lag_v2 {asset_name} panel missing columns: {missing}")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame = frame.loc[
        frame["trade_time"].notna()
        & (frame["trade_time"].dt.date == signal_day)
        & (frame["trade_time"].dt.time >= time(14, 0))
        & (frame["trade_time"].dt.time <= time(14, 29)),
        ["dt", "code", "trade_time", "last"],
    ].copy()
    if frame.empty:
        raise ValueError(
            f"parity_adjusted_stock_lag_v2 {asset_name} panel has no {signal_day} 14:00--14:29 rows"
        )
    frame["code"] = frame["code"].astype(str).str.strip().str.upper()
    frame["last"] = pd.to_numeric(frame["last"], errors="coerce")
    frame = frame.dropna(subset=["dt", "code", "trade_time", "last"])
    frame = frame.loc[frame["code"] != ""]
    if frame.empty:
        raise ValueError(f"parity_adjusted_stock_lag_v2 {asset_name} tail prices are all invalid")

    grouped = frame.sort_values(["code", "trade_time"], kind="mergesort").groupby("code", sort=False)
    result = grouped.agg(
        dt=("dt", "last"),
        start_last=("last", "first"),
        end_last=("last", "last"),
    ).reset_index()
    result["tail_return"] = result["end_last"] / result["start_last"] - 1.0
    result = result.loc[
        np.isfinite(pd.to_numeric(result["tail_return"], errors="coerce"))
        & (pd.to_numeric(result["start_last"], errors="coerce") > 0.0),
        ["dt", "code", "tail_return"],
    ].copy()
    if result.empty:
        raise ValueError(f"parity_adjusted_stock_lag_v2 {asset_name} tail returns are all invalid")
    return result


def _previous_daily_base(
    base: pd.DataFrame | None,
    *,
    signal_day: date,
    max_kappa: float,
) -> pd.DataFrame:
    if base is None or base.empty:
        raise ValueError("parity_adjusted_stock_lag_v2 missing daily_base context")
    required = {"trade_date", "code", "exchange_code", "stock_code", "conv_value", "cb_close_price"}
    missing = sorted(required.difference(base.columns))
    if missing:
        raise KeyError(f"parity_adjusted_stock_lag_v2 daily_base missing columns: {missing}")

    frame = base.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame = frame.loc[frame["trade_date"] < signal_day].copy()
    if frame.empty:
        raise ValueError(
            f"parity_adjusted_stock_lag_v2 has no daily_base observation before {signal_day}"
        )
    previous_day = frame["trade_date"].max()
    frame = frame.loc[frame["trade_date"] == previous_day].copy()
    frame["bond_code"] = (
        frame["code"].astype(str).str.strip().str.replace(r"\\.0$", "", regex=True).str.zfill(6)
        + "."
        + frame["exchange_code"].map(_normalize_exchange)
    )
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    frame["conv_value"] = pd.to_numeric(frame["conv_value"], errors="coerce")
    frame["cb_close_price"] = pd.to_numeric(frame["cb_close_price"], errors="coerce")
    frame["kappa"] = (frame["conv_value"] / frame["cb_close_price"]).clip(0.0, max_kappa)
    frame = frame.replace({"bond_code": {"": pd.NA}, "stock_code": {"": pd.NA}})
    frame = frame.dropna(subset=["bond_code", "stock_code", "kappa"])
    frame = frame.loc[np.isfinite(frame["kappa"]) & (frame["cb_close_price"] > 0.0)]
    frame = frame.drop_duplicates(subset=["bond_code"], keep="last")
    if frame.empty:
        raise ValueError(
            f"parity_adjusted_stock_lag_v2 has no usable T-1 mapping/parity on {previous_day}"
        )
    return frame[["bond_code", "stock_code", "kappa"]]


@FactorRegistry.register("parity_adjusted_stock_lag_v2")
class ParityAdjustedStockLagV2Factor(Factor):
    """Pool-free tail gap: ``kappa(T-1) * stock_tail(T) - cbond_tail(T)``."""

    name = "parity_adjusted_stock_lag_v2"
    requires_stock_panel = True

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        params = dict(params or {})
        base_source = str(params.get("daily_source", "market_cbond.daily_base")).strip()
        if not base_source:
            raise ValueError("parity_adjusted_stock_lag_v2 daily_source must not be empty")
        return [
            DailyFactorRequirement(
                source=base_source,
                columns=(
                    "exchange_code",
                    "stock_code",
                    "conv_value",
                    "cb_close_price",
                ),
                # The context may include T, but compute explicitly selects < T.
                lookback_days=2,
            )
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        signal_day = _signal_day(ctx)
        max_kappa = float(ctx.params.get("max_kappa", 2.0))
        if not np.isfinite(max_kappa) or max_kappa <= 0.0:
            raise ValueError("parity_adjusted_stock_lag_v2 max_kappa must be finite and > 0")
        base_source = str(ctx.params.get("daily_source", "market_cbond.daily_base")).strip()
        if not base_source:
            raise ValueError("parity_adjusted_stock_lag_v2 daily_source must not be empty")

        cbond_return = _tail_return(ctx.panel, signal_day=signal_day, asset_name="cbond").rename(
            columns={"code": "bond_code", "tail_return": "cbond_tail_return"}
        )
        stock_return = _tail_return(ctx.stock_panel, signal_day=signal_day, asset_name="stock").rename(
            columns={"code": "stock_code", "tail_return": "stock_tail_return"}
        )
        previous = _previous_daily_base(
            ctx.daily_data.get(base_source),
            signal_day=signal_day,
            max_kappa=max_kappa,
        )

        merged = cbond_return.merge(previous, on="bond_code", how="inner")
        merged = merged.merge(stock_return[["stock_code", "stock_tail_return"]], on="stock_code", how="inner")
        if merged.empty:
            raise ValueError(
                f"parity_adjusted_stock_lag_v2 has no matched bond/stock return rows on {signal_day}"
            )
        merged["factor_value"] = (
            merged["kappa"] * merged["stock_tail_return"] - merged["cbond_tail_return"]
        )
        merged["factor_value"] = pd.to_numeric(merged["factor_value"], errors="coerce")
        merged = merged.dropna(subset=["dt", "bond_code", "factor_value"])
        if merged.empty:
            raise ValueError(f"parity_adjusted_stock_lag_v2 values are all invalid on {signal_day}")
        index = pd.MultiIndex.from_frame(merged[["dt", "bond_code"]], names=["dt", "code"])
        output = pd.Series(merged["factor_value"].to_numpy(), index=index, dtype="float64")
        output = output[~output.index.duplicated(keep="last")]
        output.name = self.output_name(self.name)
        return output
