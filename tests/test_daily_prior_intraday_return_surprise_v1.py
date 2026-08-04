from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs.daily_prior_intraday_return_surprise_v1 import (
    DailyPriorIntradayReturnSurpriseV1Factor,
)


DT = pd.Timestamp("2025-06-30 14:30:00")
CODE = "110001.SH"


def _panel(*, add_at_cutoff_outlier: bool = False) -> pd.DataFrame:
    rows = [
        {
            "dt": DT,
            "code": CODE,
            "seq": 0,
            "trade_time": DT - pd.Timedelta(minutes=1),
            "open": 100.0,
            "last": 102.0,
        }
    ]
    if add_at_cutoff_outlier:
        rows.append(
            {
                "dt": DT,
                "code": CODE,
                "seq": 1,
                "trade_time": DT,
                "open": 1.0,
                "last": 1_000.0,
            }
        )
    return pd.DataFrame(rows).set_index(["dt", "code", "seq"])


def _daily(*, include_signal_row: bool = False, omit_date: pd.Timestamp | None = None) -> pd.DataFrame:
    dates = pd.bdate_range(end=DT.normalize(), periods=121)[:-1]
    rows: list[dict[str, object]] = []
    for index, day in enumerate(dates):
        if omit_date is not None and day == omit_date:
            continue
        ret = 0.001 if index % 2 == 0 else -0.002
        rows.append(
            {
                "trade_date": day,
                "code": "110001",
                "twap_0930_0935": 100.0,
                "twap_1430_1442": 100.0 * (1.0 + ret),
            }
        )
    if include_signal_row:
        rows.append(
            {
                "trade_date": DT.normalize(),
                "code": "110001",
                "twap_0930_0935": 1.0,
                "twap_1430_1442": 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def _compute(daily: pd.DataFrame, *, add_at_cutoff_outlier: bool = False) -> pd.Series:
    return DailyPriorIntradayReturnSurpriseV1Factor().compute(
        FactorComputeContext(
            panel=_panel(add_at_cutoff_outlier=add_at_cutoff_outlier),
            daily_data={"market_cbond.daily_twap": daily},
        )
    )


def test_return_surprise_is_registered_and_requires_prior_daily_context() -> None:
    assert (
        FactorRegistry.get("daily_prior_intraday_return_surprise_v1")
        is DailyPriorIntradayReturnSurpriseV1Factor
    )
    req = DailyPriorIntradayReturnSurpriseV1Factor.daily_requirements()
    assert req[0].source == "market_cbond.daily_twap"
    assert req[0].columns == ("twap_0930_0935", "twap_1430_1442")
    assert req[0].lookback_days == 121


def test_return_surprise_uses_current_pre_cutoff_panel_and_exactly_120_prior_sessions() -> None:
    actual = _compute(_daily(), add_at_cutoff_outlier=True)
    returns = np.array([0.001 if index % 2 == 0 else -0.002 for index in range(120)])
    expected = (0.02 - returns.mean()) / returns.std(ddof=1)
    assert actual.loc[(DT, CODE)] == pytest.approx(expected)


def test_return_surprise_ignores_same_day_and_future_daily_rows() -> None:
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
                        "twap_1430_1442": 1_000.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    actual = _compute(contaminated)
    assert actual.loc[(DT, CODE)] == pytest.approx(baseline.loc[(DT, CODE)])


def test_return_surprise_keeps_missing_prior_session_explicitly_missing() -> None:
    missing_date = pd.bdate_range(end=DT.normalize(), periods=121)[:-1][50]
    actual = _compute(_daily(omit_date=missing_date))
    assert np.isnan(actual.loc[(DT, CODE)])


def test_return_surprise_rejects_duplicate_daily_date_code_rows() -> None:
    daily = _daily()
    duplicate = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate daily date/code rows"):
        _compute(duplicate)
