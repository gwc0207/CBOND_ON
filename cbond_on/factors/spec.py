from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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

