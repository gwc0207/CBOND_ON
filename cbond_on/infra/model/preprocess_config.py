from __future__ import annotations

from typing import Any


def parse_winsor_bounds(
    raw: Any,
    *,
    default_lower: float = 0.01,
    default_upper: float = 0.99,
) -> tuple[float | None, float | None]:
    if raw is False:
        return None, None
    if raw is None or raw is True:
        return float(default_lower), float(default_upper)
    if not isinstance(raw, dict):
        return float(default_lower), float(default_upper)
    if not bool(raw.get("enabled", True)):
        return None, None

    def _parse_bound(key: str, default: float) -> float | None:
        value = raw.get(key, default)
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"", "none", "null", "nan", "false"}:
            return None
        return float(value)

    lower = _parse_bound("lower", default_lower)
    upper = _parse_bound("upper", default_upper)
    if lower is not None and lower <= 0.0:
        lower = None
    if upper is not None and upper >= 1.0:
        upper = None
    if lower is not None and not 0.0 < lower < 1.0:
        raise ValueError(f"winsor.lower must be in (0, 1), got {lower}")
    if upper is not None and not 0.0 < upper < 1.0:
        raise ValueError(f"winsor.upper must be in (0, 1), got {upper}")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(f"winsor.lower must be < winsor.upper, got {lower} >= {upper}")
    return lower, upper
