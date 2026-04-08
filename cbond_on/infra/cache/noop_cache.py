from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NoopCache:
    def get(self, key: str, default: Any = None) -> Any:
        _ = key
        return default

    def set(self, key: str, value: Any) -> None:
        _ = (key, value)

