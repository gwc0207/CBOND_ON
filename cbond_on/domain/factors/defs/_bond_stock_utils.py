from __future__ import annotations

import pandas as pd

from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import (
    _build_rebuilt_bar_frame,
    _compute_backend_runtime,
    _dataframe_backend_runtime,
    ensure_trade_time,
)


def _normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _latest_prev_bar_close(panel: pd.DataFrame) -> pd.DataFrame:
    sorted_panel = panel.sort_values("trade_time")
    frame = sorted_panel.reset_index()
    grouped = frame.groupby(["dt", "code"], sort=False)
    prev_close = pd.to_numeric(grouped["last"].nth(-2), errors="coerce")
    if "pre_close" in frame.columns:
        fallback = pd.to_numeric(grouped["pre_close"].first(), errors="coerce")
        prev_close = prev_close.reindex(fallback.index)
        prev_close = prev_close.where(prev_close.notna(), fallback)
    out = prev_close.rename("prev_bar_close").reset_index()
    out["pre_close"] = out["prev_bar_close"]
    out["code"] = _normalize_code(out["code"])
    return out


def _normalize_windowsize(ctx: FactorComputeContext) -> int:
    raw = ctx.params.get("windowsize", ctx.params.get("window_size"))
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except Exception:
        return 0


def _prepare_latest_source_panel(
    panel: pd.DataFrame,
    *,
    windowsize: int,
    backend_mode: str,
    torch_device: str,
    dataframe_backend: str,
) -> pd.DataFrame:
    panel = ensure_trade_time(panel)
    if windowsize <= 0:
        return panel
    base = (
        panel.reset_index()
        .sort_values(["dt", "code", "seq"], kind="mergesort")
        .reset_index(drop=True)
    )
    rebuilt = _build_rebuilt_bar_frame(
        base,
        window_points=windowsize,
        backend_mode=backend_mode,
        torch_device=torch_device,
        dataframe_backend=dataframe_backend,
    )
    rebuilt = rebuilt.set_index(["dt", "code", "seq"]).sort_index()
    return ensure_trade_time(rebuilt)


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
    prev_cols = {"pre_close", "prev_bar_close"}
    windowsize = _normalize_windowsize(ctx)
    backend_mode, torch_device = _compute_backend_runtime(ctx.params)
    dataframe_backend = _dataframe_backend_runtime(ctx.params)

    bond_panel = _prepare_latest_source_panel(
        ctx.panel,
        windowsize=windowsize,
        backend_mode=backend_mode,
        torch_device=torch_device,
        dataframe_backend=dataframe_backend,
    )
    missing_bond = [c for c in bond_cols if c not in bond_panel.columns and c not in prev_cols]
    if missing_bond:
        raise KeyError(f"bond panel missing columns: {missing_bond}")

    bond_latest = _latest_rows(bond_panel)
    base_cols = ["dt", "code", *[c for c in bond_cols if c in bond_latest.columns]]
    base = bond_latest[base_cols].copy()
    if any(c in bond_cols for c in prev_cols) and "prev_bar_close" not in bond_latest.columns:
        base = base.merge(_latest_prev_bar_close(bond_panel), on=["dt", "code"], how="left")
    for col in bond_cols:
        if col not in base.columns:
            base[col] = pd.NA
    base = base[["dt", "code", *bond_cols]].copy()

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

    stock_panel = _prepare_latest_source_panel(
        stock_panel,
        windowsize=windowsize,
        backend_mode=backend_mode,
        torch_device=torch_device,
        dataframe_backend=dataframe_backend,
    )
    missing_stock = [c for c in stock_cols if c not in stock_panel.columns and c not in prev_cols]
    if missing_stock:
        raise KeyError(f"stock panel missing columns: {missing_stock}")

    stock_latest = _latest_rows(stock_panel)
    if any(c in stock_cols for c in prev_cols) and "prev_bar_close" not in stock_latest.columns:
        stock_latest = stock_latest.merge(_latest_prev_bar_close(stock_panel), on=["dt", "code"], how="left")
    stock_latest = stock_latest.rename(columns={"code": "stock_code"})
    stock_latest["stock_code"] = _normalize_code(stock_latest["stock_code"])
    for col in stock_cols:
        if col not in stock_latest.columns:
            stock_latest[col] = pd.NA
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

