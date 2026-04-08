from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DateWindowCommand:
    start: date
    end: date


@dataclass(frozen=True)
class ModelScoreCommand:
    model_id: str
    start: date
    end: date
    label_cutoff: date | None = None

