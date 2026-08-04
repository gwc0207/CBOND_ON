from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_observable_seasoning_v1 as seasoning,
)
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
OTHER_BOND = "110002.SH"


def _panel() -> pd.DataFrame:
    panel = pd.DataFrame(
        [
            {"dt": DT, "code": BOND, "seq": 0, "last": 101.0},
            {"dt": DT, "code": OTHER_BOND, "seq": 0, "last": 102.0},
        ]
    ).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _daily_sources(
    *,
    include_score_and_future: bool = False,
    stale_terminal_for_bond: bool = False,
    zero_terminal_amount_for_bond: bool = False,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=64)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate((BOND, OTHER_BOND)):
        bare, exchange = code.split(".")
        for position, day in enumerate(dates):
            if stale_terminal_for_bond and code == BOND and position == len(dates) - 1:
                continue
            amount = float(1_000_000.0 + 10_000.0 * position + 500.0 * code_index)
            if zero_terminal_amount_for_bond and code == BOND and position == len(dates) - 1:
                amount = 0.0
            rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "close_price": float(100.0 + 0.1 * position + code_index),
                    "amount": amount,
                }
            )
    if include_score_and_future:
        for day in (DT.normalize(), DT.normalize() + pd.Timedelta(days=1)):
            for bare, exchange in (("110001", "SH"), ("110002", "SH")):
                rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "close_price": 9_999_999.0,
                        "amount": 9_999_999.0,
                    }
                )
    return {"market_cbond.daily_price": pd.DataFrame(rows)}


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=seasoning.KERNEL_NAME, params={"signal": entry.signal})
        for entry in seasoning.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(**source_kwargs))


def test_catalogue_daily_requirements_and_registry_are_explicit() -> None:
    entries = seasoning.factor_mining_catalog()

    assert Counter(entry.family for entry in entries) == {"prior_observable_market_seasoning": 2}
    assert set(seasoning.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(seasoning.KERNEL_NAME)
        is seasoning.FactorMiningDailyObservableSeasoningV1
    )
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in seasoning.FactorMiningDailyObservableSeasoningV1.daily_requirements()
    ] == [
        ("market_cbond.daily_price", ("exchange_code", "close_price", "amount"), 65)
    ]


def test_strict_prior_seasoning_signals_are_finite_without_infinity() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in seasoning.factor_mining_catalog()]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame["osa_terminal_amount_streak60"] == 60.0).all()
    assert (frame["osa_observation_density60"] == 1.0).all()


def test_score_day_and_future_rows_cannot_change_outputs() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_terminal_source_row_fails_closed_for_only_that_security() -> None:
    frame = _build(stale_terminal_for_bond=True)

    assert frame.loc[(DT, BOND)].isna().all()
    assert np.isfinite(frame.loc[(DT, OTHER_BOND)].to_numpy(dtype="float64")).all()


def test_observed_zero_terminal_amount_is_formula_zero_not_missing_fill() -> None:
    frame = _build(zero_terminal_amount_for_bond=True)

    assert frame.loc[(DT, BOND), "osa_terminal_amount_streak60"] == 0.0
    assert frame.loc[(DT, BOND), "osa_observation_density60"] == 1.0
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_duplicate_normalized_daily_key_is_rejected() -> None:
    sources = _daily_sources()
    duplicate = sources["market_cbond.daily_price"].iloc[[0]].copy()
    duplicate["code"] = "110001.SH"
    duplicate["exchange_code"] = "SH"
    sources["market_cbond.daily_price"] = pd.concat(
        [sources["market_cbond.daily_price"], duplicate],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        build_factor_frame(_panel(), _specs(), daily_data=sources)
