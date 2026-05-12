from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FactorContract:
    name: str
    implementation: str
    family: str = "unknown"
    params: dict[str, Any] = field(default_factory=dict)
    required_fields: tuple[str, ...] = ()
    uses_ohlc_rebuild: bool = False
    live_enabled: bool = False
    model_enabled: bool = False

