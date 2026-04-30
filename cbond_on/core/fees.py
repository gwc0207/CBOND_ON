from __future__ import annotations

from cbond_on.core.config import load_config_file


def resolve_buy_sell_bps(cfg: dict) -> tuple[float, float]:
    fee_bps = float(cfg.get("fee_bps", 0.0))
    buy_slippage_bps = float(cfg.get("buy_slippage_bps", 0.0))
    sell_slippage_bps = float(cfg.get("sell_slippage_bps", 0.0))
    # Unified overnight-style cost model:
    # buy_cost  = max(0, fee_bps - buy_slippage_bps)
    # sell_cost = max(0, fee_bps - sell_slippage_bps)
    buy_bps = max(0.0, fee_bps - buy_slippage_bps)
    sell_bps = max(0.0, fee_bps - sell_slippage_bps)
    return buy_bps, sell_bps


def load_fees_buy_sell_bps(
    *,
    config_key: str = "fees/fees",
) -> tuple[float, float, str]:
    cfg = dict(load_config_file(config_key))
    profile = str(cfg.get("profile", "overnight"))
    source = f"{config_key} profile={profile}"
    buy_bps, sell_bps = resolve_buy_sell_bps(cfg)
    return buy_bps, sell_bps, source
