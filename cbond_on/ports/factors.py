from __future__ import annotations

from typing import Any, Protocol, Sequence


class FactorEngine(Protocol):
    def compute(self, panel: Any, specs: Sequence[Any], **kwargs: Any) -> Any: ...


class FactorStore(Protocol):
    def load(self, **kwargs: Any) -> Any: ...

    def write(self, factors: Any, **kwargs: Any) -> None: ...

