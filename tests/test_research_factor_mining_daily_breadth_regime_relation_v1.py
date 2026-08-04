from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_breadth_regime_relation_v1 as breadth_relation,
)
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
OTHER_BOND = "110002.SH"
PEERS = ("110003.SH", "110004.SH")


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
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=64)
    rows: list[dict[str, object]] = []
    all_codes = (BOND, OTHER_BOND, *PEERS)
    for code_index, code in enumerate(all_codes):
        bare, exchange = code.split(".")
        for position, day in enumerate(dates):
            if stale_terminal_for_bond and code == BOND and position == len(dates) - 1:
                continue
            regime = position % 4
            positive = (code_index <= regime) if regime < 3 else True
            if code == BOND:
                positive = regime in {1, 3}
            daily_return = 0.012 + 0.001 * code_index if positive else -0.011 - 0.001 * code_index
            rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "prev_close_price": 100.0,
                    "close_price": float(100.0 * np.exp(daily_return)),
                }
            )
    if include_score_and_future:
        for day in (DT.normalize(), DT.normalize() + pd.Timedelta(days=1)):
            for code in all_codes:
                bare, exchange = code.split(".")
                rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "prev_close_price": 1.0,
                        "close_price": 9_999_999.0,
                    }
                )
    return {"market_cbond.daily_price": pd.DataFrame(rows)}


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=breadth_relation.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in breadth_relation.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(**source_kwargs))


def test_catalogue_daily_requirements_and_registry_are_explicit() -> None:
    entries = breadth_relation.factor_mining_catalog()

    assert Counter(entry.family for entry in entries) == {
        "prior_market_breadth_regime_relation": 1
    }
    assert set(breadth_relation.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(breadth_relation.KERNEL_NAME)
        is breadth_relation.FactorMiningDailyBreadthRegimeRelationV1
    )
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in breadth_relation.FactorMiningDailyBreadthRegimeRelationV1.daily_requirements()
    ] == [
        ("market_cbond.daily_price", ("exchange_code", "prev_close_price", "close_price"), 65)
    ]


def test_strict_prior_breadth_relation_is_finite_without_infinity() -> None:
    frame = _build()

    assert frame.columns.tolist() == ["brr_rank_low_high_spread60"]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_score_day_and_future_rows_cannot_change_outputs() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_terminal_source_row_fails_closed_for_only_that_security() -> None:
    frame = _build(stale_terminal_for_bond=True)

    assert pd.isna(frame.loc[(DT, BOND), "brr_rank_low_high_spread60"])
    assert np.isfinite(frame.loc[(DT, OTHER_BOND), "brr_rank_low_high_spread60"])


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
