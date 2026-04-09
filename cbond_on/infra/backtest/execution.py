from __future__ import annotations

import pandas as pd


def apply_twap_bps(price: pd.Series, bps: float, *, side: str) -> pd.Series:
    if bps == 0:
        return price
    adj = bps / 10000.0
    if side == "buy":
        return price * (1.0 + adj)
    if side == "sell":
        return price * (1.0 - adj)
    raise ValueError(f"unknown side: {side}")
