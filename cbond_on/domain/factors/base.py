from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class DailyFactorRequirement:
    source: str
    columns: tuple[str, ...] = ()
    lookback_days: int = 1


@dataclass
class FactorComputeContext:
    panel: pd.DataFrame
    stock_panel: pd.DataFrame | None = None
    bond_stock_map: pd.DataFrame | None = None
    daily_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    cache_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class Factor:
    name: str = ""
    output_col: Optional[str] = None
    requires_stock_panel: bool = False
    requires_bond_stock_map: bool = False
    requires_daily_sources: tuple[str, ...] = ()
    requires_daily_columns: dict[str, tuple[str, ...]] = {}
    daily_lookback_days: int = 1

    @classmethod
    def daily_requirements(cls, params: Dict[str, Any] | None = None) -> list[DailyFactorRequirement]:
        del params
        raw_sources = tuple(str(x) for x in getattr(cls, "requires_daily_sources", ()) if str(x))
        if not raw_sources:
            return []
        raw_columns = getattr(cls, "requires_daily_columns", {}) or {}
        lookback = int(getattr(cls, "daily_lookback_days", 1) or 1)
        if lookback <= 0:
            lookback = 1
        out: list[DailyFactorRequirement] = []
        for source in raw_sources:
            cols_raw = raw_columns.get(source, ())
            cols = tuple(str(c) for c in cols_raw if str(c))
            out.append(
                DailyFactorRequirement(
                    source=source,
                    columns=cols,
                    lookback_days=lookback,
                )
            )
        return out

    def __init__(self, *, name: Optional[str] = None, output_col: Optional[str] = None):
        if name:
            self.name = name
        if output_col:
            self.output_col = output_col

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        raise NotImplementedError

    def output_name(self, default: str) -> str:
        return self.output_col or default


def ensure_panel_index(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must use a MultiIndex (dt, code, seq)")
    expected = ("dt", "code", "seq")
    if panel.index.names[:3] != expected:
        panel = panel.copy()
        panel.index.set_names(expected, inplace=True)
    return panel


def panel_groupby_keys(panel: pd.DataFrame) -> Iterable[int | str]:
    return panel.index.names[:2]

