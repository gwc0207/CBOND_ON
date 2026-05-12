from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ReportWriter(Protocol):
    def write(self, payload: Any, *, out_dir: Path, **kwargs: Any) -> Path: ...

