from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cbond_on.infra.benchmark import service
from cbond_on.infra.benchmark.service import BenchmarkPoolConfig


def _pool_cfg() -> BenchmarkPoolConfig:
    return BenchmarkPoolConfig(
        method="strict_official_prev_close_split",
        pool_table="",
        buy_twap_col="twap_buy",
        sell_twap_col="twap_sell",
        use_window_data=False,
        window_data_root="",
        min_price=0.0,
        max_price=999.0,
        positive_field="",
        positive_fallback_field="",
        positive_threshold=0.0,
        pool_lag_trading_days=1,
        pool_asset="cbond",
        use_pool_weight=False,
    )


def test_strict_cycle_detail_compounds_buy_and_sell_legs(monkeypatch) -> None:
    buy_holdings = pd.DataFrame(
        {
            "code": ["113001"],
            "weight": [1.0],
            "buy_trade_day": [pd.Timestamp("2026-07-08")],
            "buy_price": [100.0],
            "buy_close_price": [110.0],
            "buy_leg_ret_gross": [0.10],
            "buy_leg_ret_net": [0.10],
            "weighted_buy_leg_ret_net": [0.10],
            "weighted_buy_leg_ret_gross": [0.10],
            "buy_cost_bps": [0.0],
        }
    )

    def fake_sell_detail(**kwargs):
        return pd.DataFrame(
            {
                "code": ["113001"],
                "sell_trade_day": [pd.Timestamp("2026-07-09")],
                "strict_prev_close_price": [110.0],
                "twap_sell": [132.0],
                "strict_sell_price": [132.0],
                "sell_price": [132.0],
                "sell_price_source": ["twap"],
                "sell_missing_fallback": [False],
                "strict_prev_close_source": ["buy_close"],
                "strict_sell_leg_gross_ret": [0.20],
                "strict_sell_leg_net_ret": [0.20],
                "weighted_sell_leg_ret_net": [0.20],
                "weighted_sell_leg_ret_gross": [0.20],
                "strict_sell_fee_weighted": [0.0],
                "sell_cost_bps": [0.0],
            }
        )

    monkeypatch.setattr(service, "compute_strict_sell_detail_for_holdings", fake_sell_detail)

    detail = service.compute_strict_cycle_detail_for_holdings(
        raw_data_root="raw",
        buy_day=date(2026, 7, 8),
        sell_day=date(2026, 7, 9),
        buy_holdings=buy_holdings,
        sell_bps=0.0,
        pool_cfg=_pool_cfg(),
    )

    assert float(detail.loc[0, "return_net"]) == pytest.approx(0.32)
    assert float(detail.loc[0, "weighted_return"]) == pytest.approx(0.32)
