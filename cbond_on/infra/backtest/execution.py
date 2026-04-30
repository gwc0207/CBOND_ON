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


def compound_return_from_prices(buy_exec: pd.Series, sell_exec: pd.Series) -> pd.Series:
    """
    Unified overnight return math.

    Equivalent to:
        R_full = (1 + R_buy) * (1 + R_sell) - 1
    where R_buy and R_sell are two sequential legs bridged by the same close.
    Algebraically this is also sell_exec / buy_exec - 1.
    """
    return (sell_exec / buy_exec) - 1.0


def split_cycle_return_by_bridge(
    buy_exec: pd.Series,
    sell_exec: pd.Series,
    bridge_price: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Decompose full-cycle return into buy/sell legs with a shared denominator.

    buy_leg_ret_net  = (bridge - buy_exec) / buy_exec
    sell_leg_ret_net = (sell_exec - bridge) / buy_exec
    full_cycle_ret_net = sell_exec / buy_exec - 1
                     = buy_leg_ret_net + sell_leg_ret_net
    """
    buy_leg = (bridge_price - buy_exec) / buy_exec
    sell_leg = (sell_exec - bridge_price) / buy_exec
    full = compound_return_from_prices(buy_exec, sell_exec)
    return buy_leg, sell_leg, full
