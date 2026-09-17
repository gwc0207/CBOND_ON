from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_capacity_rank_coupling_v1 as capacity,
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
    stale_base_for_bond: bool = False,
    omit_latest_base_globally: bool = False,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=72)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(_MARKET_CODES):
        bare, exchange = code.split(".")
        price = 100.0 + 4.0 * code_index
        for position, day in enumerate(dates):
            daily_return = 0.012 * np.sin(0.37 * position + 0.43 * code_index)
            previous = price
            price *= float(np.exp(daily_return))
            amount = 1_000_000.0 * np.exp(
                0.47 * np.cos(0.29 * position + 0.71 * code_index)
                + 0.15 * np.sin(0.17 * position - 0.29 * code_index)
            )
            price_rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "prev_close_price": previous,
                    "close_price": price,
                    "amount": amount,
                }
            )
            if (
                (stale_base_for_bond and code == BOND and position == len(dates) - 1)
                or (omit_latest_base_globally and position == len(dates) - 1)
            ):
                continue
            base_rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "remain_size": 6.0
                    + 1.7 * code_index
                    + 0.19 * np.sin(0.23 * position + 0.53 * code_index),
                }
            )
    if include_score_and_future:
        for day in (DT.normalize(), DT.normalize() + pd.Timedelta(days=1)):
            for code in _MARKET_CODES:
                bare, exchange = code.split(".")
                price_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "prev_close_price": 1.0,
                        "close_price": 9_999_999.0,
                        "amount": 9_999_999_999.0,
                    }
                )
                base_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "remain_size": 1.0,
                    }
                )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=capacity.KERNEL_NAME, params={"signal": entry.signal})
        for entry in capacity.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(**source_kwargs))


def test_catalogue_family_daily_contract_and_registry_are_explicit() -> None:
    entries = capacity.factor_mining_catalog()

    assert Counter(entry.family for entry in entries) == {
        "prior_capacity_normalized_return_rank_coupling": 1
    }
    assert set(capacity.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(capacity.KERNEL_NAME) is capacity.FactorMiningDailyCapacityRankCouplingV1
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in capacity.FactorMiningDailyCapacityRankCouplingV1.daily_requirements()
    ] == [
        (
            "market_cbond.daily_price",
            ("exchange_code", "prev_close_price", "close_price", "amount"),
            75,
        ),
        ("market_cbond.daily_base", ("exchange_code", "remain_size"), 75),
    ]


def test_signal_is_finite_for_strict_prior_aligned_history() -> None:
    frame = _build()

    assert frame.columns.tolist() == ["prcn_return_capacity_rank_corr60"]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()


def test_score_day_and_future_rows_cannot_change_output() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_per_security_terminal_base_row_fails_closed() -> None:
    frame = _build(stale_base_for_bond=True)

    assert np.isnan(frame.loc[(DT, BOND), "prcn_return_capacity_rank_corr60"])
    assert np.isfinite(frame.loc[(DT, OTHER_BOND), "prcn_return_capacity_rank_corr60"])


def test_misaligned_global_price_and_base_anchor_fails_closed() -> None:
    frame = _build(omit_latest_base_globally=True)

    assert frame.isna().all().all()


def test_duplicate_normalized_daily_key_is_rejected() -> None:
    sources = _daily_sources()
    duplicate = sources["market_cbond.daily_base"].iloc[[0]].copy()
    duplicate["code"] = "110001.SH"
    duplicate["exchange_code"] = "SH"
    sources["market_cbond.daily_base"] = pd.concat(
        [sources["market_cbond.daily_base"], duplicate],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        build_factor_frame(_panel(), _specs(), daily_data=sources)
