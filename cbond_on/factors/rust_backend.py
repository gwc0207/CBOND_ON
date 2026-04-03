from __future__ import annotations

from importlib import import_module
from time import perf_counter
from typing import Sequence

import pandas as pd

from cbond_on.factors.base import ensure_panel_index
from cbond_on.factors.spec import FactorSpec, build_factor_col


def _import_rust_module():
    try:
        return import_module("cbond_on_rust")
    except Exception as exc:
        raise RuntimeError(
            "factor engine=rust but module 'cbond_on_rust' is not available. "
            "Build/install the Rust extension first (see rust/factor_engine/README.md)."
        ) from exc


def _specs_payload(specs: Sequence[FactorSpec]) -> list[dict]:
    payload: list[dict] = []
    for spec in specs:
        payload.append(
            {
                "name": str(spec.name),
                "factor": str(spec.factor),
                "params": dict(spec.params or {}),
                "output_col": spec.output_col,
            }
        )
    return payload


def build_factor_frame_rust(
    panel: pd.DataFrame,
    specs: Sequence[FactorSpec],
    *,
    stock_panel: pd.DataFrame | None = None,
    bond_stock_map: pd.DataFrame | None = None,
    compute_backend_params: dict | None = None,
) -> pd.DataFrame:
    if not specs:
        return pd.DataFrame()

    t_total = perf_counter()
    t_prepare_panel = perf_counter()
    panel = ensure_panel_index(panel)
    panel_df = panel.reset_index().copy()
    panel_df = panel_df.sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(drop=True)
    panel_df["dt"] = pd.to_datetime(panel_df["dt"], errors="coerce")
    panel_df["code"] = panel_df["code"].astype(str)
    panel_df["trade_time"] = pd.to_datetime(panel_df["trade_time"], errors="coerce")
    panel_df["__trade_time_ns__"] = panel_df["trade_time"].astype("int64")
    t_prepare_panel = perf_counter() - t_prepare_panel

    t_prepare_context = perf_counter()
    stock_df = None
    if stock_panel is not None and not stock_panel.empty:
        stock_df = ensure_panel_index(stock_panel).reset_index().copy()
        stock_df = stock_df.sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(drop=True)
        stock_df["dt"] = pd.to_datetime(stock_df["dt"], errors="coerce")
        stock_df["code"] = stock_df["code"].astype(str)
        stock_df["trade_time"] = pd.to_datetime(stock_df["trade_time"], errors="coerce")
        stock_df["__trade_time_ns__"] = stock_df["trade_time"].astype("int64")

    map_df = None
    if bond_stock_map is not None and not bond_stock_map.empty:
        map_df = bond_stock_map.copy()
    t_prepare_context = perf_counter() - t_prepare_context

    t_payload = perf_counter()
    payload = _specs_payload(specs)
    t_payload = perf_counter() - t_payload
    module = _import_rust_module()
    if not hasattr(module, "compute_factor_frame"):
        raise RuntimeError("module 'cbond_on_rust' missing function: compute_factor_frame")

    t_rust = perf_counter()
    out = module.compute_factor_frame(
        panel_df,
        payload,
        stock_df,
        map_df,
        dict(compute_backend_params or {}),
    )
    t_rust = perf_counter() - t_rust
    if not isinstance(out, pd.DataFrame):
        raise RuntimeError("cbond_on_rust.compute_factor_frame must return pandas.DataFrame")
    if "dt" not in out.columns or "code" not in out.columns:
        raise RuntimeError("cbond_on_rust output must include columns: dt, code")

    t_post = perf_counter()
    out = out.copy()
    out["dt"] = pd.to_datetime(out["dt"], errors="coerce")
    out["code"] = out["code"].astype(str)
    out = out.set_index(["dt", "code"]).sort_index()

    cols = [build_factor_col(spec) for spec in specs]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise RuntimeError(f"cbond_on_rust output missing factor columns: {missing}")
    t_post = perf_counter() - t_post
    print(
        "rust_engine:",
        f"panel_rows={len(panel_df)}",
        f"specs={len(specs)}",
        f"t_prepare_panel={t_prepare_panel:.2f}s",
        f"t_prepare_context={t_prepare_context:.2f}s",
        f"t_payload={t_payload:.2f}s",
        f"t_rust_call={t_rust:.2f}s",
        f"t_post={t_post:.2f}s",
        f"total={perf_counter() - t_total:.2f}s",
        flush=True,
    )
    return out[cols]
