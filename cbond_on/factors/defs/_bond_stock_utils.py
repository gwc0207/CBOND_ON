from __future__ import annotations

import pandas as pd

from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time


def _normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _latest_rows(panel: pd.DataFrame) -> pd.DataFrame:
    latest = (
        panel.sort_values("trade_time")
        .groupby(level=["dt", "code"], sort=False)
        .tail(1)
        .reset_index(level="seq", drop=True)
        .reset_index()
    )
    latest["code"] = _normalize_code(latest["code"])
    return latest


def build_bond_stock_latest_frame(
    ctx: FactorComputeContext,
    *,
    bond_cols: list[str],
    stock_cols: list[str],
) -> pd.DataFrame:
    bond_panel = ensure_trade_time(ctx.panel)
    missing_bond = [c for c in bond_cols if c not in bond_panel.columns]
    if missing_bond:
        raise KeyError(f"bond panel missing columns: {missing_bond}")

    bond_latest = _latest_rows(bond_panel)
    base = bond_latest[["dt", "code", *bond_cols]].copy()

    mapping = ctx.bond_stock_map
    if mapping is None or mapping.empty:
        base["stock_code"] = pd.NA
    else:
        if "code" not in mapping.columns or "stock_code" not in mapping.columns:
            raise KeyError("bond_stock_map missing columns: ['code', 'stock_code']")
        mp = mapping[["code", "stock_code"]].copy()
        mp["code"] = _normalize_code(mp["code"])
        mp["stock_code"] = _normalize_code(mp["stock_code"])
        mp = mp.replace({"code": {"": pd.NA}, "stock_code": {"": pd.NA}})
        mp = mp.dropna(subset=["code", "stock_code"]).drop_duplicates(subset=["code"], keep="last")
        base = base.merge(mp, on="code", how="left")

    if not stock_cols:
        return base

    stock_panel = ctx.stock_panel
    if stock_panel is None or stock_panel.empty:
        for col in stock_cols:
            base[f"stock_{col}"] = pd.NA
        return base

    stock_panel = ensure_trade_time(stock_panel)
    missing_stock = [c for c in stock_cols if c not in stock_panel.columns]
    if missing_stock:
        raise KeyError(f"stock panel missing columns: {missing_stock}")

    stock_latest = _latest_rows(stock_panel)
    stock_latest = stock_latest.rename(columns={"code": "stock_code"})
    stock_latest["stock_code"] = _normalize_code(stock_latest["stock_code"])
    stock_latest = stock_latest[["dt", "stock_code", *stock_cols]].rename(
        columns={col: f"stock_{col}" for col in stock_cols}
    )

    merged = base.merge(stock_latest, on=["dt", "stock_code"], how="left")
    merged = merged.drop_duplicates(subset=["dt", "code"], keep="last")
    return merged


def to_dt_code_series(frame: pd.DataFrame, values: pd.Series, *, name: str) -> pd.Series:
    idx = pd.MultiIndex.from_frame(frame[["dt", "code"]], names=["dt", "code"])
    out = pd.Series(values.to_numpy(), index=idx)
    out = out[~out.index.duplicated(keep="last")]
    out = out.fillna(0.0)
    out.name = name
    return out

