from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_relative_rank_tail_contradiction_v1 as tail,
)
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
OTHER_BOND = "110002.SH"
_MARKET_CODES = (BOND, OTHER_BOND, "110003.SH", "110004.SH", "110005.SH")


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
    stale_for_bond: bool = False,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=72)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(_MARKET_CODES):
        bare, exchange = code.split(".")
        price = 100.0 + 4.0 * code_index
        for position, day in enumerate(dates):
            if stale_for_bond and code == BOND and position == len(dates) - 1:
                continue
            daily_return = 0.012 * np.sin(0.43 * position + 0.69 * code_index)
            previous = price
            price *= float(np.exp(daily_return))
            amount = 1_000_000.0 * np.exp(
                0.62 * np.sin(0.31 * position + 1.13 * code_index)
                + 0.17 * np.cos(0.27 * position - 0.41 * code_index)
            )
            rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "prev_close_price": previous,
                    "close_price": price,
                    "amount": amount,
                }
            )
    if include_score_and_future:
        for day in (DT.normalize(), DT.normalize() + pd.Timedelta(days=1)):
            for code in _MARKET_CODES:
                bare, exchange = code.split(".")
                rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "prev_close_price": 1.0,
                        "close_price": 9_999_999.0,
                        "amount": 9_999_999_999.0,
                    }
                )
    return {"market_cbond.daily_price": pd.DataFrame(rows)}


def _build(**source_kwargs: object) -> pd.DataFrame:
    spec = FactorSpec(name=tail._SIGNAL, factor=tail.KERNEL_NAME, params={"signal": tail._SIGNAL})
    return build_factor_frame(_panel(), [spec], daily_data=_daily_sources(**source_kwargs))


def test_catalogue_daily_contract_and_registry_are_explicit() -> None:
    entries = tail.factor_mining_catalog()

    assert [(entry.family, entry.signal) for entry in entries] == [
        ("prior_relative_return_flow_tail_contradiction", tail._SIGNAL)
    ]
    assert set(tail.FORMULAS) == {tail._SIGNAL}
    assert FactorRegistry.get(tail.KERNEL_NAME) is tail.FactorMiningDailyRelativeRankTailContradictionV1
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in tail.FactorMiningDailyRelativeRankTailContradictionV1.daily_requirements()
    ] == [
        ("market_cbond.daily_price", ("exchange_code", "prev_close_price", "close_price", "amount"), 75)
    ]


def test_tail_signal_is_finite_for_strict_prior_history() -> None:
    frame = _build()

    assert frame.columns.tolist() == [tail._SIGNAL]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()


def test_score_day_and_future_rows_cannot_change_tail_state() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_or_invalid_terminal_row_fails_closed() -> None:
    stale = _build(stale_for_bond=True)
    assert np.isnan(stale.loc[(DT, BOND), tail._SIGNAL])
    assert np.isfinite(stale.loc[(DT, OTHER_BOND), tail._SIGNAL])

    sources = _daily_sources()
    daily = sources["market_cbond.daily_price"]
    latest = daily["trade_date"].max()
    row = daily.index[(daily["code"] == "110001") & (daily["trade_date"] == latest)][0]
    daily.loc[row, "amount"] = 0.0
    invalid = build_factor_frame(
        _panel(),
        [FactorSpec(name=tail._SIGNAL, factor=tail.KERNEL_NAME, params={"signal": tail._SIGNAL})],
        daily_data=sources,
    )
    assert np.isnan(invalid.loc[(DT, BOND), tail._SIGNAL])
    assert not np.isinf(invalid.to_numpy(dtype="float64")).any()


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
        _build_with_sources(sources)


def _build_with_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spec = FactorSpec(name=tail._SIGNAL, factor=tail.KERNEL_NAME, params={"signal": tail._SIGNAL})
    return build_factor_frame(_panel(), [spec], daily_data=sources)
