from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_liquidity_channel_composition_v1 as composition,
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
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=72)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate((BOND, OTHER_BOND)):
        bare, exchange = code.split(".")
        for position, day in enumerate(dates):
            if stale_terminal_for_bond and code == BOND and position == len(dates) - 1:
                continue
            rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "volume": float(np.exp(9.0 + 0.33 * np.sin(0.39 * position + code_index))),
                    "amount": float(np.exp(12.0 + 0.41 * np.sin(0.67 * position + 0.2 * code_index))),
                    "deal": float(np.exp(6.0 + 0.29 * np.cos(0.53 * position + 0.5 * code_index))),
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
                        "volume": 9_999_999.0,
                        "amount": 9_999_999.0,
                        "deal": 9_999_999.0,
                    }
                )
    return {"market_cbond.daily_price": pd.DataFrame(rows)}


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=composition.KERNEL_NAME, params={"signal": entry.signal})
        for entry in composition.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(**source_kwargs))


def test_catalogue_daily_requirements_and_registry_are_explicit() -> None:
    entries = composition.factor_mining_catalog()

    assert Counter(entry.family for entry in entries) == {
        "prior_liquidity_channel_composition": 5
    }
    assert set(composition.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(composition.KERNEL_NAME)
        is composition.FactorMiningDailyLiquidityChannelCompositionV1
    )
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in composition.FactorMiningDailyLiquidityChannelCompositionV1.daily_requirements()
    ] == [
        ("market_cbond.daily_price", ("exchange_code", "volume", "amount", "deal"), 75)
    ]


def test_strict_prior_composition_signals_are_finite_without_infinity() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in composition.factor_mining_catalog()]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_score_day_and_future_rows_cannot_change_outputs() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_terminal_source_row_fails_closed_for_only_that_security() -> None:
    frame = _build(stale_terminal_for_bond=True)

    assert frame.loc[(DT, BOND)].isna().all()
    assert np.isfinite(frame.loc[(DT, OTHER_BOND)].to_numpy(dtype="float64")).all()


def test_nonpositive_terminal_deal_fails_closed_without_infinity() -> None:
    sources = _daily_sources()
    price = sources["market_cbond.daily_price"]
    row = price.index[(price["code"] == "110001") & (price["trade_date"] == price["trade_date"].max())][0]
    price.loc[row, "deal"] = 0.0

    frame = build_factor_frame(_panel(), _specs(), daily_data=sources)

    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame.loc[(DT, BOND)].isna().all()


def test_duplicate_normalized_daily_key_is_rejected() -> None:
    sources = _daily_sources()
    duplicate = sources["market_cbond.daily_price"].iloc[[0]].copy()
    duplicate["code"] = "110001.SH"
    duplicate["exchange_code"] = "SH"
    sources["market_cbond.daily_price"] = pd.concat(
        [sources["market_cbond.daily_price"], duplicate], ignore_index=True
    )

    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        build_factor_frame(_panel(), _specs(), daily_data=sources)
