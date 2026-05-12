from __future__ import annotations

from typing import Any, Protocol


class SelectionStrategy(Protocol):
    def select(self, scores: Any, *args: Any, **kwargs: Any) -> Any: ...


def select_positions(strategy: SelectionStrategy, scores: Any, *args: Any, **kwargs: Any) -> Any:
    return strategy.select(scores, *args, **kwargs)

