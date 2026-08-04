from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_ohlc_wick_path_asymmetry_v1 as ohlc,
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
    stale_for_bond: bool = False,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=72)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate((BOND, OTHER_BOND)):
        bare, exchange = code.split(".")
        price = 100.0 + 7.0 * code_index
        for position, day in enumerate(dates):
            if stale_for_bond and code == BOND and position == len(dates) - 1:
                continue
            gap = 0.006 * np.sin(0.29 * position + 0.17 * code_index)
            intraday = 0.011 * np.sin(0.53 * position + 0.43 * code_index)
            previous = price
            opened = previous * float(np.exp(gap))
            close = opened * float(np.exp(intraday))
            high = max(opened, close) * (1.0 + 0.006 + 0.002 * (position % 3))
            low = min(opened, close) * (1.0 - 0.005 - 0.001 * (position % 2))
            price = close
            rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "prev_close_price": previous,
                    "open_price": opened,
                    "high_price": high,
                    "low_price": low,
                    "close_price": close,
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
                        "prev_close_price": 1.0,
                        "open_price": 1.0,
                        "high_price": 9_999_999.0,
                        "low_price": 0.1,
                        "close_price": 9_999_999.0,
                    }
                )
    return {"market_cbond.daily_price": pd.DataFrame(rows)}


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=ohlc.KERNEL_NAME, params={"signal": entry.signal})
        for entry in ohlc.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(**source_kwargs))


def test_catalogue_family_daily_contract_and_registry_are_explicit() -> None:
    entries = ohlc.factor_mining_catalog()

    assert Counter(entry.family for entry in entries) == {
        "prior_daily_ohlc_wick_path_asymmetry": 2
    }
    assert set(ohlc.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(ohlc.KERNEL_NAME) is ohlc.FactorMiningDailyOhlcWickPathAsymmetryV1
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in ohlc.FactorMiningDailyOhlcWickPathAsymmetryV1.daily_requirements()
    ] == [
        (
            "market_cbond.daily_price",
            (
                "exchange_code",
                "prev_close_price",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
            ),
            75,
        )
    ]


def test_signals_are_finite_for_strict_prior_history() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in ohlc.factor_mining_catalog()]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()


def test_score_day_and_future_rows_cannot_change_outputs() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_terminal_source_row_fails_closed_for_both_signals() -> None:
    frame = _build(stale_for_bond=True)

    assert frame.loc[(DT, BOND)].isna().all()
    assert np.isfinite(frame.loc[(DT, OTHER_BOND)].to_numpy(dtype="float64")).all()


def test_invalid_terminal_ohlc_state_fails_closed_without_inf() -> None:
    sources = _daily_sources()
    daily = sources["market_cbond.daily_price"]
    latest = daily["trade_date"].max()
    row = daily.index[(daily["code"] == "110001") & (daily["trade_date"] == latest)][0]
    daily.loc[row, "high_price"] = daily.loc[row, "low_price"] * 0.5

    frame = build_factor_frame(_panel(), _specs(), daily_data=sources)

    assert frame.loc[(DT, BOND)].isna().all()
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
