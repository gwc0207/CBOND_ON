from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

from cbond_on.app.usecases.live_runtime import run_once


def execute(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
    stage_reporter: Callable[[str], None] | None = None,
) -> Path:
    return run_once(
        start=start,
        target=target,
        mode=mode,
        stage_reporter=stage_reporter,
    )
