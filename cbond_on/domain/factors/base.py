from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import pandas as pd


@dataclass
class FactorComputeContext:
    panel: pd.DataFrame
    stock_panel: pd.DataFrame | None = None
    bond_stock_map: pd.DataFrame | None = None
    params: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    cache_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class Factor:
    name: str = ""
    output_col: Optional[str] = None
    requires_stock_panel: bool = False
    requires_bond_stock_map: bool = False

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

