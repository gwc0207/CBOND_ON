from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from cbond_on.strategies.base import Strategy


@dataclass
class _StrategyRegistry:
    _items: dict[str, type[Strategy]] = field(default_factory=dict)

    def register(self, strategy_id: str) -> Callable[[type[Strategy]], type[Strategy]]:
        key = str(strategy_id).strip()
        if not key:
            raise ValueError("strategy_id must not be empty")

        def _decorator(cls: type[Strategy]) -> type[Strategy]:
            if key in self._items:
                raise ValueError(f"duplicate strategy registration: {key}")
            self._items[key] = cls
            return cls

        return _decorator

    def get(self, strategy_id: str) -> Strategy:
        key = str(strategy_id).strip()
        if key not in self._items:
            raise KeyError(f"unknown strategy_id: {key}")
        return self._items[key]()

    def names(self) -> Iterable[str]:
        return self._items.keys()


StrategyRegistry = _StrategyRegistry()

