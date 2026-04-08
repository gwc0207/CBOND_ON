from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageResult:
    stage: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class BacktestResult:
    out_dir: Path
    days: int

