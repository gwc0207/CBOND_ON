from __future__ import annotations

from cbond_on.core.config import load_config_file


def resolve_buy_sell_bps(cfg: dict) -> tuple[float, float]:
    fee_bps = float(cfg.get("fee_bps", 0.0))
    if "buy_bps" in cfg or "sell_bps" in cfg:
        buy_bps = float(cfg.get("buy_bps", cfg.get("twap_bps", 0.0))) + fee_bps
        sell_bps = float(cfg.get("sell_bps", cfg.get("twap_bps", 0.0))) + fee_bps
    else:
        twap_bps = float(cfg.get("twap_bps", 0.0))
        buy_bps = twap_bps + fee_bps
        sell_bps = twap_bps + fee_bps
    return max(0.0, buy_bps), max(0.0, sell_bps)


def load_fees_buy_sell_bps(
    *,
    config_key: str = "fees/fees",
) -> tuple[float, float, str]:
    cfg = dict(load_config_file(config_key))
    source = config_key
    buy_bps, sell_bps = resolve_buy_sell_bps(cfg)
    return buy_bps, sell_bps, source
