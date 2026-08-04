from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_return_liquidity_topology_v1 as topology,
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
            daily_return = 0.012 * np.sin(0.73 * position + 0.31 * code_index)
            previous = price
            price *= float(np.exp(daily_return))
            amount = 1_000_000.0 * np.exp(
                0.35 * np.cos(0.51 * position + 0.19 * code_index)
                + 0.09 * np.sin(0.91 * position)
            )
            deal = 7_000.0 * np.exp(
                0.27 * np.sin(0.59 * position + 0.11 * code_index)
                + 0.07 * np.cos(0.37 * position)
            )
            rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "prev_close_price": previous,
                    "close_price": price,
                    "amount": amount,
                    "deal": deal,
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
                        "close_price": 9_999_999.0,
                        "amount": 9_999_999_999.0,
                        "deal": 9_999_999.0,
                    }
                )
    return {"market_cbond.daily_price": pd.DataFrame(rows)}


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=topology.KERNEL_NAME, params={"signal": entry.signal})
        for entry in topology.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(**source_kwargs))


def test_catalogue_families_and_daily_contract_are_registered() -> None:
    entries = topology.factor_mining_catalog()

    assert Counter(entry.family for entry in entries) == {
        "prior_return_liquidity_information_dependence": 2,
        "prior_joint_return_liquidity_state_topology": 1,
    }
    assert set(topology.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(topology.KERNEL_NAME) is topology.FactorMiningDailyReturnLiquidityTopologyV1
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in topology.FactorMiningDailyReturnLiquidityTopologyV1.daily_requirements()
    ] == [
        (
            "market_cbond.daily_price",
            ("exchange_code", "prev_close_price", "close_price", "amount", "deal"),
            75,
        )
    ]


def test_signals_are_finite_for_strict_prior_history() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in topology.factor_mining_catalog()]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()


def test_score_day_and_future_rows_cannot_change_outputs() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_latest_source_row_fails_closed_for_that_security() -> None:
    frame = _build(stale_for_bond=True)

    assert frame.loc[(DT, BOND)].isna().all()
    assert np.isfinite(frame.loc[(DT, OTHER_BOND)].to_numpy(dtype="float64")).all()


def test_nonpositive_latest_deal_only_invalidates_deal_dependent_outputs() -> None:
    sources = _daily_sources()
    daily = sources["market_cbond.daily_price"]
    latest = daily["trade_date"].max()
    index = daily.index[(daily["code"] == "110001") & (daily["trade_date"] == latest)][0]
    daily.loc[index, "deal"] = 0.0

    frame = build_factor_frame(_panel(), _specs(), daily_data=sources)

    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame.loc[(DT, BOND), "rlmi_return_deal_sign_mutual_information60"] != frame.loc[
        (DT, BOND), "rlmi_return_deal_sign_mutual_information60"
    ]
    assert frame.loc[(DT, BOND), "rlmi_return_trade_size_sign_mutual_information60"] != frame.loc[
        (DT, BOND), "rlmi_return_trade_size_sign_mutual_information60"
    ]
    assert np.isfinite(frame.loc[(DT, BOND), "rjst_amount_joint_transition_entropy60"])


def test_missing_terminal_calendar_adjacency_fails_closed() -> None:
    returns = np.linspace(-0.02, 0.02, num=45, dtype="float64")
    log_liquidity = np.linspace(10.0, 11.0, num=45, dtype="float64")
    sessions = np.arange(45, dtype="int64")
    sessions[-1] += 1

    assert np.isnan(topology._sign_mutual_information(returns, log_liquidity, sessions))
    assert np.isnan(topology._joint_transition_entropy(returns, log_liquidity, sessions))


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
