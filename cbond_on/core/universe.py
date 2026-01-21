from __future__ import annotations

import pandas as pd


def filter_tradable(
    df: pd.DataFrame,
    *,
    buy_twap_col: str,
    sell_twap_col: str | None = None,
    min_amount: float = 0.0,
    min_volume: float = 0.0,
) -> pd.DataFrame:
    if buy_twap_col not in df.columns:
        raise KeyError(f"missing {buy_twap_col}")
    mask = df[buy_twap_col].notna() & (df[buy_twap_col] > 0)
    if sell_twap_col:
        if sell_twap_col not in df.columns:
            raise KeyError(f"missing {sell_twap_col}")
        mask = mask & df[sell_twap_col].notna() & (df[sell_twap_col] > 0)
    if "amount" in df.columns and min_amount > 0:
        mask = mask & (df["amount"] >= min_amount)
    if "volume" in df.columns and min_volume > 0:
        mask = mask & (df["volume"] >= min_volume)
    return df[mask]
