from __future__ import annotations

import pandas as pd


def normalize_price_bound(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return 0.0
    return float(value)


def filter_tradable(
    df: pd.DataFrame,
    *,
    buy_twap_col: str,
    sell_twap_col: str | None = None,
    min_amount: float = 0.0,
    min_volume: float = 0.0,
    min_price: float = 0.0,
    max_price: float = 0.0,
) -> pd.DataFrame:
    if buy_twap_col not in df.columns:
        raise KeyError(f"missing {buy_twap_col}")
    min_price_value = normalize_price_bound(min_price)
    max_price_value = normalize_price_bound(max_price)
    buy_price = pd.to_numeric(df[buy_twap_col], errors="coerce")
    mask = buy_price.notna() & (buy_price > 0)
    if min_price_value > 0:
        mask = mask & (buy_price >= min_price_value)
    if max_price_value > 0:
        mask = mask & (buy_price <= max_price_value)
    if sell_twap_col:
        if sell_twap_col not in df.columns:
            raise KeyError(f"missing {sell_twap_col}")
        sell_price = pd.to_numeric(df[sell_twap_col], errors="coerce")
        mask = mask & sell_price.notna() & (sell_price > 0)
    if "amount" in df.columns and min_amount > 0:
        mask = mask & (df["amount"] >= min_amount)
    if "volume" in df.columns and min_volume > 0:
        mask = mask & (df["volume"] >= min_volume)
    return df[mask]
