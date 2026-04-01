from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Iterable

import pandas as pd

from cbond_on.factors.base import FactorComputeContext, ensure_panel_index
from cbond_on.factors.spec import FactorSpec, build_factor_col


@dataclass
class FactorUpdateResult:
    written: int = 0
    skipped: int = 0


def _collect_windowsize_plan(specs: list[FactorSpec]) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for spec in specs:
        params = spec.params if isinstance(spec.params, dict) else {}
        raw = params.get("windowsize", params.get("window_size"))
        if raw is None:
            continue
        try:
            size = int(raw)
        except Exception as exc:
            raise ValueError(f"factor {spec.name} has invalid windowsize: {raw!r}") from exc
        if size <= 0:
            raise ValueError(f"factor {spec.name} windowsize must be > 0, got {size}")
        if size in seen:
            continue
        seen.add(size)
        values.append(size)
    values = sorted(values)
    if len(values) > 8:
        raise ValueError(
            f"too many unique windowsize values ({len(values)}), max allowed is 8: {values}"
        )
    return values


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


def _spawn_spec_context(
    base_ctx: FactorComputeContext,
    spec: FactorSpec,
    *,
    windowsize_plan: list[int],
) -> FactorComputeContext:
    params = dict(base_ctx.params) if isinstance(base_ctx.params, dict) else {}
    if isinstance(spec.params, dict):
        params.update(spec.params)
    panel = base_ctx.panel.copy(deep=False)
    panel.attrs = dict(getattr(base_ctx.panel, "attrs", {}) or {})
    panel.attrs["__factor_spec_name__"] = str(spec.name)
    panel.attrs["__factor_kernel_name__"] = str(spec.factor)
    params["__factor_kernel_name__"] = str(spec.factor)
    params["__factor_spec_name__"] = str(spec.name)
    params["__factor_kernel_params__"] = dict(spec.params) if isinstance(spec.params, dict) else {}
    if windowsize_plan:
        params["__ohlc_windows_plan__"] = list(windowsize_plan)
    return FactorComputeContext(
        panel=panel,
        stock_panel=base_ctx.stock_panel,
        bond_stock_map=base_ctx.bond_stock_map,
        params=params,
        cache=base_ctx.cache,
        cache_lock=base_ctx.cache_lock,
    )


def _compute_from_shared_context(
    base_ctx: FactorComputeContext,
    spec: FactorSpec,
    *,
    windowsize_plan: list[int],
) -> pd.Series:
    t0 = perf_counter()
    print(f"[factor:{spec.name}] start", flush=True)
    factor = spec.build()
    ctx = _spawn_spec_context(base_ctx, spec, windowsize_plan=windowsize_plan)
    series = factor.compute(ctx)
    if not isinstance(series, pd.Series):
        raise ValueError(f"factor {spec.name} must return a Series")
    series.name = build_factor_col(spec)
    non_na = int(series.notna().sum()) if hasattr(series, "notna") else -1
    print(
        f"[factor:{spec.name}] done elapsed={perf_counter() - t0:.2f}s non_na={non_na}",
        flush=True,
    )
    return series


def build_factor_frame(
    panel: pd.DataFrame,
    specs: Iterable[FactorSpec],
    *,
    stock_panel: pd.DataFrame | None = None,
    bond_stock_map: pd.DataFrame | None = None,
    workers: int = 1,
    compute_backend_params: dict | None = None,
) -> pd.DataFrame:
    spec_list = list(specs)
    if not spec_list:
        return pd.DataFrame()
    windowsize_plan = _collect_windowsize_plan(spec_list)

    panel = ensure_panel_index(panel).copy(deep=False)
    if stock_panel is not None and not stock_panel.empty:
        stock_panel = ensure_panel_index(stock_panel)
    runtime_backend = {}
    if isinstance(compute_backend_params, dict):
        raw_backend = compute_backend_params.get("__compute_backend__")
        if isinstance(raw_backend, dict):
            runtime_backend = dict(raw_backend)
    panel.attrs["__compute_backend__"] = runtime_backend
    base_ctx = FactorComputeContext(
        panel=panel,
        stock_panel=stock_panel,
        bond_stock_map=bond_stock_map,
        params=dict(compute_backend_params or {}),
    )

    workers = max(1, int(workers))
    if workers <= 1 or len(spec_list) <= 1:
        series_list: list[pd.Series] = []
        for spec in spec_list:
            series_list.append(
                _compute_from_shared_context(
                    base_ctx,
                    spec,
                    windowsize_plan=windowsize_plan,
                )
            )
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
                windowsize_plan=windowsize_plan,
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

