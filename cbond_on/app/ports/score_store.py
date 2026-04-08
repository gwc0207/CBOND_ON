from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd


class ScoreStorePort(Protocol):
    def load_by_date(self, root: Path) -> dict[date, pd.DataFrame]: ...

    def write_by_date(self, root: Path, frame: pd.DataFrame, *, overwrite: bool = False, dedupe: bool = True) -> None: ...

