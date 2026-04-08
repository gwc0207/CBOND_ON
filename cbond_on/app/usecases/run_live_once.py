from __future__ import annotations

from datetime import date
from pathlib import Path

from cbond_on.services.live.live_service import run_once


def execute(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> Path:
    return run_once(
        start=start,
        target=target,
        mode=mode,
    )

