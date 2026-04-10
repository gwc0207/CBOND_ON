from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from cbond_on.core.registry import FactorRegistry


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
class FactorContextRequirements:
    stock_panel_required: bool = False
    bond_stock_map_required: bool = False
    stock_panel_factors: tuple[str, ...] = ()
    bond_stock_map_factors: tuple[str, ...] = ()


def infer_factor_context_requirements(specs: Sequence[FactorSpec]) -> FactorContextRequirements:
    stock_factors: list[str] = []
    map_factors: list[str] = []
    for spec in specs:
        cls = FactorRegistry.get(spec.factor)
        if bool(getattr(cls, "requires_stock_panel", False)):
            stock_factors.append(str(spec.name))
        if bool(getattr(cls, "requires_bond_stock_map", False)):
            map_factors.append(str(spec.name))
    return FactorContextRequirements(
        stock_panel_required=bool(stock_factors),
        bond_stock_map_required=bool(map_factors),
        stock_panel_factors=tuple(stock_factors),
        bond_stock_map_factors=tuple(map_factors),
    )

