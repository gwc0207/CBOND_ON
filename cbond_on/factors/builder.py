from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from cbond_on.factors.base import FactorComputeContext, ensure_panel_index
from cbond_on.factors.spec import FactorSpec, build_factor_col


@dataclass
class FactorUpdateResult:
    written: int = 0
    skipped: int = 0


def compute_factor_from_panel(panel: pd.DataFrame, spec: FactorSpec) -> pd.Series:
    panel = ensure_panel_index(panel)
    factor = spec.build()
    ctx = FactorComputeContext(panel=panel, params=spec.params)
    series = factor.compute(ctx)
    if not isinstance(series, pd.Series):
        raise ValueError(f"factor {spec.name} must return a Series")
    series.name = build_factor_col(spec)
    return series


def build_factor_frame(
    panel: pd.DataFrame,
    specs: Iterable[FactorSpec],
) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    for spec in specs:
        series_list.append(compute_factor_from_panel(panel, spec))
    if not series_list:
        return pd.DataFrame()
    out = pd.concat(series_list, axis=1)
    return out

