from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs.parity_adjusted_stock_lag_v2 import ParityAdjustedStockLagV2Factor


def _panel(*, code: str, prices: list[float]) -> pd.DataFrame:
    dt = pd.Timestamp("2026-07-30 14:30:00")
    return pd.DataFrame(
        {
            "dt": [dt, dt, dt],
            "code": [code, code, code],
            "seq": [0, 1, 2],
            "trade_time": [
                pd.Timestamp("2026-07-30 14:00:00"),
                pd.Timestamp("2026-07-30 14:29:00"),
                # This post-cutoff observation must not affect the factor.
                pd.Timestamp("2026-07-30 14:30:00"),
            ],
            "last": prices,
        }
    ).set_index(["dt", "code", "seq"])


def _context(*, base: pd.DataFrame | None = None) -> FactorComputeContext:
    bond = _panel(code="110001.SH", prices=[100.0, 102.0, 999.0])
    bond.attrs["__build_day__"] = "2026-07-30"
    stock = _panel(code="600000.SH", prices=[10.0, 10.5, 999.0])
    if base is None:
        base = pd.DataFrame(
            {
                "trade_date": [date(2026, 7, 29), date(2026, 7, 30)],
                "code": ["110001", "110001"],
                "exchange_code": ["SH", "SH"],
                "stock_code": ["600000", "600000"],
                # The current-day value is deliberately very different.
                "conv_value": [120.0, 200.0],
                "cb_close_price": [100.0, 100.0],
            }
        )
    return FactorComputeContext(
        panel=bond,
        stock_panel=stock,
        daily_data={"market_cbond.daily_base": base},
        params={"daily_source": "market_cbond.daily_base", "max_kappa": 2.0},
    )


def test_v2_is_registered_and_has_no_pool_dependency():
    assert FactorRegistry.get("parity_adjusted_stock_lag_v2") is ParityAdjustedStockLagV2Factor
    requirements = ParityAdjustedStockLagV2Factor.daily_requirements()
    assert [item.source for item in requirements] == ["market_cbond.daily_base"]
    assert requirements[0].lookback_days == 2


def test_v2_uses_tminus1_parity_and_excludes_post_cutoff_tick():
    out = ParityAdjustedStockLagV2Factor().compute(_context())

    # kappa(T-1)=120/100=1.2, stock tail=5%, bond tail=2%, so 4%.
    # Using same-day parity would give 8%, and using 14:30 prices would be nonsensical.
    assert out.index.tolist() == [(pd.Timestamp("2026-07-30 14:30:00"), "110001.SH")]
    assert out.iloc[0] == pytest.approx(0.04)


def test_v2_fails_fast_when_daily_schema_is_incomplete():
    base = pd.DataFrame(
        {
            "trade_date": [date(2026, 7, 29)],
            "code": ["110001"],
            "exchange_code": ["SH"],
            "stock_code": ["600000"],
            "conv_value": [120.0],
        }
    )
    with pytest.raises(KeyError, match="daily_base missing columns"):
        ParityAdjustedStockLagV2Factor().compute(_context(base=base))
