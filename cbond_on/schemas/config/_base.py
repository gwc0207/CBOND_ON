from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when a runtime config misses a required contract."""


def require_mapping(cfg: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        raise ConfigValidationError(f"{name} must be a mapping")
    return dict(cfg)


def require_keys(cfg: Mapping[str, Any], *, name: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in cfg or cfg.get(key) in (None, "")]
    if missing:
        raise ConfigValidationError(f"{name} missing required keys: {missing}")

