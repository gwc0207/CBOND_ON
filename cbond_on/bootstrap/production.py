from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class LiveRequest:
    start: str | date | None = None
    target: str | date | None = None
    mode: str = "default"


def build_live_request(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> LiveRequest:
    return LiveRequest(start=start, target=target, mode=mode)

