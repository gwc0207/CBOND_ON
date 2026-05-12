from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TradeListSummary:
    trade_date: date | str
    rows: int
    notional: float | None = None

