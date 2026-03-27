from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd

from cbond_on.factors.base import FactorComputeContext, ensure_panel_index
from cbond_on.factors.spec import FactorSpec, build_factor_col


@dataclass
class FactorUpdateResult:
    written: int = 0
    skipped: int = 0


def compute_factor_from_panel(panel: pd.DataFrame, spec: FactorSpec) -> pd.Series:
    panel = ensure_panel_index(panel)
    factor = spec.build()
    ctx = FactorComputeContext(panel=panel, params=spec.params)
    series = factor.compute(ctx)
    if not isinstance(series, pd.Series):
        raise ValueError(f"factor {spec.name} must return a Series")
    series.name = build_factor_col(spec)
    return series


def compute_factor_from_context(
    panel: pd.DataFrame,
    spec: FactorSpec,
    *,
    stock_panel: pd.DataFrame | None = None,
    bond_stock_map: pd.DataFrame | None = None,
) -> pd.Series:
    panel = ensure_panel_index(panel)
    if stock_panel is not None and not stock_panel.empty:
        stock_panel = ensure_panel_index(stock_panel)
    factor = spec.build()
    ctx = FactorComputeContext(
        panel=panel,
        stock_panel=stock_panel,
        bond_stock_map=bond_stock_map,
        params=spec.params,
    )
    series = factor.compute(ctx)
    if not isinstance(series, pd.Series):
        raise ValueError(f"factor {spec.name} must return a Series")
    series.name = build_factor_col(spec)
    return series


def _spawn_spec_context(base_ctx: FactorComputeContext, spec: FactorSpec) -> FactorComputeContext:
    return FactorComputeContext(
        panel=base_ctx.panel,
        stock_panel=base_ctx.stock_panel,
        bond_stock_map=base_ctx.bond_stock_map,
        params=spec.params,
        cache=base_ctx.cache,
        cache_lock=base_ctx.cache_lock,
    )


def _compute_from_shared_context(base_ctx: FactorComputeContext, spec: FactorSpec) -> pd.Series:
    factor = spec.build()
    ctx = _spawn_spec_context(base_ctx, spec)
    series = factor.compute(ctx)
    if not isinstance(series, pd.Series):
        raise ValueError(f"factor {spec.name} must return a Series")
    series.name = build_factor_col(spec)
    return series


def build_factor_frame(
    panel: pd.DataFrame,
    specs: Iterable[FactorSpec],
    *,
    stock_panel: pd.DataFrame | None = None,
    bond_stock_map: pd.DataFrame | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    spec_list = list(specs)
    if not spec_list:
        return pd.DataFrame()

    panel = ensure_panel_index(panel)
    if stock_panel is not None and not stock_panel.empty:
        stock_panel = ensure_panel_index(stock_panel)
    base_ctx = FactorComputeContext(
        panel=panel,
        stock_panel=stock_panel,
        bond_stock_map=bond_stock_map,
        params={},
    )

    workers = max(1, int(workers))
    if workers <= 1 or len(spec_list) <= 1:
        series_list: list[pd.Series] = []
        for spec in spec_list:
            series_list.append(_compute_from_shared_context(base_ctx, spec))
        out = pd.concat(series_list, axis=1)
        return out

    max_workers = min(workers, len(spec_list))
    series_list: list[pd.Series] = []
    ordered: dict[int, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _compute_from_shared_context,
                base_ctx,
                spec,
            ): (idx, spec)
            for idx, spec in enumerate(spec_list)
        }
        for fut in as_completed(futures):
            idx, spec = futures[fut]
            try:
                ordered[idx] = fut.result()
            except Exception as exc:
                raise RuntimeError(f"factor compute failed: {spec.name}") from exc
    for idx in range(len(spec_list)):
        series_list.append(ordered[idx])
    out = pd.concat(series_list, axis=1)
    return out

