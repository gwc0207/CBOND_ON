from __future__ import annotations

from typing import Any, Protocol


class ReadyGate(Protocol):
    def check(self, **kwargs: Any) -> None: ...


class TradeWriter(Protocol):
    def write(self, trade_list: Any, **kwargs: Any) -> None: ...

