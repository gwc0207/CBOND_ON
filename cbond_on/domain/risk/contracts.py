from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class RiskModelError(ValueError):
    """Raised when a CB-Risk input or estimation contract is not satisfied."""


@dataclass(frozen=True)
class FactorDefinition:
    """Definition of one cross-sectional convertible-bond style exposure."""

    name: str
    source_columns: tuple[str, ...]
    transform: str = "identity"
    min_coverage: float = 0.8
    orthogonalize_against: tuple[str, ...] = ()
    orthogonalize_square: bool = False


def _as_columns(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        columns = (value.strip(),)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        columns = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        raise RiskModelError(f"risk factor {name!r} source_columns must be a string or list")
    if not columns:
        raise RiskModelError(f"risk factor {name!r} has no source columns")
    return columns


def factor_definitions_from_config(raw: Any) -> tuple[FactorDefinition, ...]:
    """Parse the compact config representation without coupling it to infra."""

    if not isinstance(raw, list) or not raw:
        raise RiskModelError("risk.factors must be a non-empty list")

    definitions: list[FactorDefinition] = []
    names: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise RiskModelError("each risk factor definition must be an object")
        name = str(item.get("name") or "").strip()
        if not name:
            raise RiskModelError("risk factor missing name")
        if name in names:
            raise RiskModelError(f"duplicate risk factor: {name}")
        names.add(name)
        against = item.get("orthogonalize_against") or ()
        if isinstance(against, str):
            against = (against,)
        if not isinstance(against, Iterable) or isinstance(against, (bytes, bytearray, Mapping)):
            raise RiskModelError(f"risk factor {name!r} orthogonalize_against must be a string or list")
        definitions.append(
            FactorDefinition(
                name=name,
                source_columns=_as_columns(item.get("source_columns"), name=name),
                transform=str(item.get("transform") or "identity").strip().lower(),
                min_coverage=float(item.get("min_coverage", 0.8)),
                orthogonalize_against=tuple(str(value).strip() for value in against if str(value).strip()),
                orthogonalize_square=bool(item.get("orthogonalize_square", False)),
            )
        )
    return tuple(definitions)
