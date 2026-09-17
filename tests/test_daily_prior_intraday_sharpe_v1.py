from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.operators.daily_prior_intraday_sharpe_v1 import (
    DailyPriorIntradaySharpeV1Factor,
)


DT = pd.Timestamp("2025-06-30 14:30:00")
CODE = "110001.SH"


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dt": DT,
                "code": CODE,
                "seq": 0,
                "trade_time": DT - pd.Timedelta(minutes=1),
            }
        ]
    ).set_index(["dt", "code", "seq"])


def _daily(*, include_signal_row: bool = False, omit_date: pd.Timestamp | None = None) -> pd.DataFrame:
    dates = pd.bdate_range(end=DT.normalize(), periods=121)[:-1]
    rows: list[dict[str, object]] = []
    for index, day in enumerate(dates):
        if omit_date is not None and day == omit_date:
            continue
        ret = 0.002 if index % 2 == 0 else -0.001
        rows.append(
            {
                "trade_date": day,
                "code": "110001",
                "twap_0930_0935": 100.0,
                "twap_1442_1457": 100.0 * (1.0 + ret),
            }
        )
    if include_signal_row:
        rows.append(
            {
                "trade_date": DT.normalize(),
                "code": "110001",
                "twap_0930_0935": 1.0,
                "twap_1442_1457": 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def _compute(daily: pd.DataFrame) -> pd.Series:
    return DailyPriorIntradaySharpeV1Factor().compute(
        FactorComputeContext(panel=_panel(), daily_data={"market_cbond.daily_twap": daily})
    )


def test_prior_intraday_sharpe_is_registered_and_requests_current_plus_prior_sessions() -> None:
    assert FactorRegistry.get("daily_prior_intraday_sharpe_v1") is DailyPriorIntradaySharpeV1Factor
    req = DailyPriorIntradaySharpeV1Factor.daily_requirements()
    assert len(req) == 1
    assert req[0].source == "market_cbond.daily_twap"
    assert req[0].columns == ("twap_0930_0935", "twap_1442_1457")
    assert req[0].lookback_days == 121


def test_prior_intraday_sharpe_uses_exactly_120_strictly_prior_sessions() -> None:
    actual = _compute(_daily())
    returns = np.array([0.002 if index % 2 == 0 else -0.001 for index in range(120)])
    expected = returns.mean() / returns.std(ddof=1)
    assert actual.loc[(DT, CODE)] == pytest.approx(expected)


def test_prior_intraday_sharpe_is_unchanged_by_same_day_or_future_twap_rows() -> None:
    baseline = _compute(_daily())
    contaminated = _daily(include_signal_row=True)
    contaminated = pd.concat(
        [
            contaminated,
            pd.DataFrame(
                [
                    {
                        "trade_date": DT.normalize() + pd.Timedelta(days=1),
                        "code": "110001",
                        "twap_0930_0935": 1.0,
                        "twap_1442_1457": 1_000.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    actual = _compute(contaminated)
    assert actual.loc[(DT, CODE)] == pytest.approx(baseline.loc[(DT, CODE)])


def test_prior_intraday_sharpe_keeps_a_missing_prior_session_explicitly_missing() -> None:
    missing_date = pd.bdate_range(end=DT.normalize(), periods=121)[:-1][50]
    actual = _compute(_daily(omit_date=missing_date))
    assert np.isnan(actual.loc[(DT, CODE)])


def test_prior_intraday_sharpe_rejects_duplicate_daily_date_code_rows() -> None:
    daily = _daily()
    duplicate = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate daily date/code rows"):
        _compute(duplicate)
