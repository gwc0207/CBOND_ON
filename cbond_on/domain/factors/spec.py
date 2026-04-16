from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement


@dataclass
class FactorSpec:
    name: str
    factor: str
    params: Dict[str, Any] = field(default_factory=dict)
    output_col: Optional[str] = None

    def build(self):
        cls = FactorRegistry.get(self.factor)
        return cls(name=self.name, output_col=self.output_col)


def build_factor_col(spec: FactorSpec) -> str:
    return spec.output_col or spec.name


@dataclass(frozen=True)
class FactorDailyContextRequirement:
    source: str
    columns: tuple[str, ...] = ()
    lookback_days: int = 1
    factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactorContextRequirements:
    stock_panel_required: bool = False
    bond_stock_map_required: bool = False
    stock_panel_factors: tuple[str, ...] = ()
    bond_stock_map_factors: tuple[str, ...] = ()
    daily_required: bool = False
    daily_requirements: tuple[FactorDailyContextRequirement, ...] = ()
    daily_factors: tuple[str, ...] = ()


def _normalize_daily_requirement(
    raw: object,
    *,
    default_lookback: int,
) -> DailyFactorRequirement | None:
    if isinstance(raw, DailyFactorRequirement):
        source = str(raw.source or "").strip()
        if not source:
            return None
        cols = tuple(str(c).strip() for c in raw.columns if str(c).strip())
        lookback = int(raw.lookback_days or default_lookback)
        if lookback <= 0:
            lookback = default_lookback
        if lookback <= 0:
            lookback = 1
        return DailyFactorRequirement(source=source, columns=cols, lookback_days=lookback)

    if isinstance(raw, str):
        source = raw.strip()
        if not source:
            return None
        lookback = max(1, int(default_lookback or 1))
        return DailyFactorRequirement(source=source, columns=(), lookback_days=lookback)

    if isinstance(raw, dict):
        source = str(raw.get("source", "")).strip()
        if not source:
            return None
        columns_raw = raw.get("columns", ())
        if isinstance(columns_raw, str):
            columns = (columns_raw.strip(),) if columns_raw.strip() else ()
        else:
            columns = tuple(str(c).strip() for c in (columns_raw or ()) if str(c).strip())
        lookback = int(raw.get("lookback_days", default_lookback) or default_lookback)
        if lookback <= 0:
            lookback = default_lookback
        if lookback <= 0:
            lookback = 1
        return DailyFactorRequirement(source=source, columns=columns, lookback_days=lookback)

    return None


def infer_factor_context_requirements(specs: Sequence[FactorSpec]) -> FactorContextRequirements:
    stock_factors: list[str] = []
    map_factors: list[str] = []

    daily_source_columns: dict[str, set[str]] = {}
    daily_source_factors: dict[str, set[str]] = {}
    daily_source_lookback: dict[str, int] = {}

    for spec in specs:
        cls = FactorRegistry.get(spec.factor)
        if bool(getattr(cls, "requires_stock_panel", False)):
            stock_factors.append(str(spec.name))
        if bool(getattr(cls, "requires_bond_stock_map", False)):
            map_factors.append(str(spec.name))

        default_lookback = int(getattr(cls, "daily_lookback_days", 1) or 1)
        if default_lookback <= 0:
            default_lookback = 1

        raw_reqs: list[object] = []
        if hasattr(cls, "daily_requirements"):
            try:
                req_items = cls.daily_requirements(dict(spec.params or {}))
            except TypeError:
                req_items = cls.daily_requirements()  # type: ignore[misc]
            if req_items is not None:
                raw_reqs = list(req_items)

        for raw in raw_reqs:
            req = _normalize_daily_requirement(raw, default_lookback=default_lookback)
            if req is None:
                continue
            source = req.source
            if source not in daily_source_columns:
                daily_source_columns[source] = set()
            daily_source_columns[source].update(req.columns)

            if source not in daily_source_factors:
                daily_source_factors[source] = set()
            daily_source_factors[source].add(str(spec.name))

            prev = daily_source_lookback.get(source, 1)
            daily_source_lookback[source] = max(prev, int(req.lookback_days or 1))

    daily_reqs: list[FactorDailyContextRequirement] = []
    for source in sorted(daily_source_columns):
        cols = tuple(sorted(daily_source_columns[source]))
        factors = tuple(sorted(daily_source_factors.get(source, set())))
        lookback = max(1, int(daily_source_lookback.get(source, 1)))
        daily_reqs.append(
            FactorDailyContextRequirement(
                source=source,
                columns=cols,
                lookback_days=lookback,
                factors=factors,
            )
        )

    daily_factor_set: set[str] = set()
    for req in daily_reqs:
        daily_factor_set.update(req.factors)

    return FactorContextRequirements(
        stock_panel_required=bool(stock_factors),
        bond_stock_map_required=bool(map_factors),
        stock_panel_factors=tuple(stock_factors),
        bond_stock_map_factors=tuple(map_factors),
        daily_required=bool(daily_reqs),
        daily_requirements=tuple(daily_reqs),
        daily_factors=tuple(sorted(daily_factor_set)),
    )
