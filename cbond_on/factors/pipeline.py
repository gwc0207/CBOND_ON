from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from cbond_on.core.utils import progress
from cbond_on.data.panel import read_panel_data
from cbond_on.factors.builder import build_factor_frame
from cbond_on.factors.spec import FactorSpec, build_factor_col
from cbond_on.factors.storage import FactorStore


@dataclass
class FactorPipelineResult:
    written: int = 0
    skipped: int = 0


def _iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current = current + pd.Timedelta(days=1)


def run_factor_pipeline(
    panel_data_root: str | Path,
    factor_data_root: str | Path,
    start: date,
    end: date,
    *,
    window_minutes: int = 15,
    panel_name: str | None = None,
    overwrite: bool = False,
    full_refresh: bool = False,
    specs: Sequence[FactorSpec],
) -> FactorPipelineResult:
    result = FactorPipelineResult()
    panel_data_root = Path(panel_data_root)
    store = FactorStore(Path(factor_data_root), panel_name=panel_name, window_minutes=window_minutes)

    total_days = (end - start).days + 1
    for day in progress(_iter_dates(start, end), desc="build_factors", unit="day", total=total_days):
        panel = read_panel_data(
            panel_data_root,
            day,
            window_minutes=window_minutes,
            panel_name=panel_name,
        ).data
        if panel is None or panel.empty:
            continue

        existing = pd.DataFrame()
        if not overwrite and not full_refresh:
            existing = store.read_day(day)

        if existing.empty or overwrite or full_refresh:
            to_compute = specs
        else:
            existing_cols = set(existing.columns)
            to_compute = [s for s in specs if build_factor_col(s) not in existing_cols]

        if not to_compute:
            result.skipped += 1
            continue

        new_frame = build_factor_frame(panel, to_compute)
        if new_frame.empty:
            continue

        if existing.empty or overwrite or full_refresh:
            merged = new_frame
        else:
            merged = existing.join(new_frame, how="outer")
        store.write_day(day, merged)
        result.written += 1

    return result

