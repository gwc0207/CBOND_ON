from __future__ import annotations

import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * 0.0
    return (series - series.mean()) / std


def rank_pct(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.rank(pct=True)

