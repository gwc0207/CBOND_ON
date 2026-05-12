from __future__ import annotations

from typing import Any, Protocol


class TradeListBuilder(Protocol):
    def build(self, positions: Any, *args: Any, **kwargs: Any) -> Any: ...


def build_trade_list(builder: TradeListBuilder, positions: Any, *args: Any, **kwargs: Any) -> Any:
    return builder.build(positions, *args, **kwargs)

