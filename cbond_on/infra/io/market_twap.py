from __future__ import annotations

from datetime import date

import pandas as pd

from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.infra.data.io import read_table_range


def iter_open_days(raw_root: str, start: date, end: date) -> list[date]:
    return list_trading_days_from_raw(raw_root, start, end, kind="snapshot", asset="cbond")


def read_twap_daily(raw_root: str, day: date) -> pd.DataFrame:
    df = read_table_range(raw_root, "market_cbond.daily_twap", day, day)
    if df.empty:
        return df
    if "instrument_code" in df.columns and "exchange_code" in df.columns:
        df = df.copy()
        df["code"] = df["instrument_code"].astype(str) + "." + df["exchange_code"].astype(str)
    return df

