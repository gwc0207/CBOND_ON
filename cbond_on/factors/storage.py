from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from cbond_on.core.naming import make_window_label


@dataclass
class FactorStore:
    root: Path
    panel_name: Optional[str] = None
    window_minutes: int = 15

    def day_path(self, day: date) -> Path:
        label = self.panel_name or make_window_label(self.window_minutes)
        month = f"{day.year:04d}-{day.month:02d}"
        filename = f"{day.strftime('%Y%m%d')}.parquet"
        return self.root / "factors" / label / month / filename

    def read_day(self, day: date) -> pd.DataFrame:
        path = self.day_path(day)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def write_day(self, day: date, df: pd.DataFrame) -> None:
        path = self.day_path(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=True)

