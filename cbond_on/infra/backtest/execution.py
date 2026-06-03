from __future__ import annotations

import pandas as pd


def bps_to_rate(bps: float) -> float:
    return max(0.0, float(bps)) / 10000.0


def apply_twap_bps(price: pd.Series, bps: float, *, side: str) -> pd.Series:
    if bps == 0:
        return price
    adj = bps_to_rate(bps)
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
    Decompose full-cycle return into sequential buy/sell attribution legs.

    buy_leg_ret_net  = bridge / buy_exec - 1
    sell_leg_ret_net = sell_exec / bridge - 1
    full_cycle_ret_net = sell_exec / buy_exec - 1
                     = (1 + buy_leg_ret_net) * (1 + sell_leg_ret_net) - 1
    """
    buy_leg = (bridge_price / buy_exec) - 1.0
    sell_leg = (sell_exec / bridge_price) - 1.0
    full = compound_return_from_prices(buy_exec, sell_exec)
    return buy_leg, sell_leg, full


def split_cycle_return_by_bridge_with_cost(
    buy_price: pd.Series,
    sell_price: pd.Series,
    bridge_price: pd.Series,
    *,
    buy_bps: float,
    sell_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Strict full-overnight net return with buy/sell costs.

    Main return:
        full_net = sell_price * (1 - sell_cost_rate)
                   / (buy_price * (1 + buy_cost_rate)) - 1

    Legs are attribution fields only:
        buy_leg_net  = bridge / (buy_price * (1 + buy_cost_rate)) - 1
        sell_leg_net = sell_price * (1 - sell_cost_rate) / bridge - 1

    Do not add the two legs to recover the main return; use compounding:
        full_net = (1 + buy_leg_net) * (1 + sell_leg_net) - 1
    """
    buy_raw = pd.to_numeric(buy_price, errors="coerce")
    sell_raw = pd.to_numeric(sell_price, errors="coerce")
    bridge = pd.to_numeric(bridge_price, errors="coerce")
    buy_rate = bps_to_rate(buy_bps)
    sell_rate = bps_to_rate(sell_bps)
    buy_exec = buy_raw * (1.0 + buy_rate)
    sell_exec = sell_raw * (1.0 - sell_rate)

    buy_leg = (bridge / buy_exec) - 1.0
    sell_leg = (sell_exec / bridge) - 1.0
    full = (sell_exec / buy_exec) - 1.0
    return buy_leg, sell_leg, full


def apply_cost_to_full_cycle_return(
    gross_return: pd.Series,
    *,
    buy_bps: float,
    sell_bps: float,
) -> pd.Series:
    """
    Apply the strict full-overnight net cost convention.

    gross_return is interpreted as sell_price / buy_price - 1.
    """
    ret = pd.to_numeric(gross_return, errors="coerce")
    buy_rate = bps_to_rate(buy_bps)
    sell_rate = bps_to_rate(sell_bps)
    return ((1.0 + ret) * (1.0 - sell_rate) / (1.0 + buy_rate)) - 1.0
