from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_relative_rank_coupling_v1 as rank_coupling,
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
        price = 100.0 + 3.0 * code_index
        for position, day in enumerate(dates):
            if stale_for_bond and code == BOND and position == len(dates) - 1:
                continue
            daily_return = 0.011 * np.sin(0.41 * position + 0.73 * code_index)
            previous = price
            price *= float(np.exp(daily_return))
            # Code-specific waves deliberately rotate the date-local amount ranks.
            amount = 1_000_000.0 * np.exp(
                0.55 * np.sin(0.37 * position + 1.21 * code_index)
                + 0.23 * np.cos(0.19 * position - 0.47 * code_index)
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
            for bare, exchange in (tuple(code.split(".")) for code in _MARKET_CODES):
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


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=rank_coupling.KERNEL_NAME, params={"signal": entry.signal})
        for entry in rank_coupling.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(**source_kwargs))


def test_catalogue_and_daily_contract_are_registered() -> None:
    entries = rank_coupling.factor_mining_catalog()

    assert Counter(entry.family for entry in entries) == {
        "prior_relative_return_amount_rank_coupling": 1,
    }
    assert set(rank_coupling.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(rank_coupling.KERNEL_NAME) is rank_coupling.FactorMiningDailyRelativeRankCouplingV1
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in rank_coupling.FactorMiningDailyRelativeRankCouplingV1.daily_requirements()
    ] == [
        (
            "market_cbond.daily_price",
            ("exchange_code", "prev_close_price", "close_price", "amount"),
            75,
        )
    ]


def test_signal_is_finite_for_strict_prior_history() -> None:
    frame = _build()

    assert frame.columns.tolist() == ["drrc_return_amount_rank_spearman60"]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()


def test_score_day_and_future_rows_cannot_change_outputs() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_latest_source_row_fails_closed_for_that_security() -> None:
    frame = _build(stale_for_bond=True)

    assert np.isnan(frame.loc[(DT, BOND), "drrc_return_amount_rank_spearman60"])
    assert np.isfinite(frame.loc[(DT, OTHER_BOND), "drrc_return_amount_rank_spearman60"])


def test_nonpositive_latest_amount_fails_closed() -> None:
    sources = _daily_sources()
    daily = sources["market_cbond.daily_price"]
    latest = daily["trade_date"].max()
    index = daily.index[(daily["code"] == "110001") & (daily["trade_date"] == latest)][0]
    daily.loc[index, "amount"] = 0.0

    frame = build_factor_frame(_panel(), _specs(), daily_data=sources)

    assert np.isnan(frame.loc[(DT, BOND), "drrc_return_amount_rank_spearman60"])
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
