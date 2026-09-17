from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1 as rank_concordance,
)
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
OTHER_BOND = "110002.SH"
_MARKET_CODES = (BOND, OTHER_BOND, "110003.SH", "110004.SH", "110005.SH")
_STOCK_CODES = ("600001", "600002", "600003", "600004", "600001")


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
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=72)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    stock_prices = {stock: 15.0 + number for number, stock in enumerate(sorted(set(_STOCK_CODES)))}
    bond_prices = {code: 100.0 + 2.0 * number for number, code in enumerate(_MARKET_CODES)}
    for position, day in enumerate(dates):
        stock_returns = {
            stock: 0.018 * np.sin(
                0.37 * position + 2.0 * np.pi * number / len(set(_STOCK_CODES))
            )
            for number, stock in enumerate(sorted(set(_STOCK_CODES)))
        }
        stock_previous = dict(stock_prices)
        for stock, value in stock_returns.items():
            stock_prices[stock] *= float(np.exp(value))
        for code_index, (code, stock_code) in enumerate(zip(_MARKET_CODES, _STOCK_CODES, strict=True)):
            bare, exchange = code.split(".")
            stock_return = stock_returns[stock_code]
            bond_return = (
                0.52 * stock_return
                + 0.007 * np.sin(0.83 * position + 0.29 * code_index)
                + 0.002 * code_index
            )
            bond_previous = bond_prices[code]
            bond_prices[code] *= float(np.exp(bond_return))
            price_rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "prev_close_price": bond_previous,
                    "close_price": bond_prices[code],
                }
            )
            if not (stale_base_for_bond and code == BOND and position == len(dates) - 1):
                base_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "stock_code": stock_code,
                        "stk_prev_close_price": stock_previous[stock_code],
                        "stk_close_price": stock_prices[stock_code],
                    }
                )
    if include_score_and_future:
        for day in (DT.normalize(), DT.normalize() + pd.Timedelta(days=1)):
            for code, stock_code in zip(_MARKET_CODES, _STOCK_CODES, strict=True):
                bare, exchange = code.split(".")
                price_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "prev_close_price": 1.0,
                        "close_price": 9_999_999.0,
                    }
                )
                base_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "stock_code": stock_code,
                        "stk_prev_close_price": 1.0,
                        "stk_close_price": 9_999_999.0,
                    }
                )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=rank_concordance.KERNEL_NAME, params={"signal": entry.signal})
        for entry in rank_concordance.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(**source_kwargs))


def test_catalogue_daily_requirements_and_registry_are_explicit() -> None:
    entries = rank_concordance.factor_mining_catalog()

    assert Counter(entry.family for entry in entries) == {
        "prior_bond_stock_cross_sectional_rank_concordance": 3
    }
    assert set(rank_concordance.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(rank_concordance.KERNEL_NAME)
        is rank_concordance.FactorMiningDailyBondStockCrossSectionalRankConcordanceV1
    )
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in rank_concordance.FactorMiningDailyBondStockCrossSectionalRankConcordanceV1.daily_requirements()
    ] == [
        ("market_cbond.daily_price", ("exchange_code", "prev_close_price", "close_price"), 75),
        (
            "market_cbond.daily_base",
            ("exchange_code", "stock_code", "stk_prev_close_price", "stk_close_price"),
            75,
        ),
    ]


def test_strict_prior_rank_signals_are_finite() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in rank_concordance.factor_mining_catalog()]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()


def test_score_day_and_future_rows_cannot_change_outputs() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_terminal_base_row_fails_closed_for_only_that_security() -> None:
    frame = _build(stale_base_for_bond=True)

    assert frame.loc[(DT, BOND)].isna().all()
    assert np.isfinite(frame.loc[(DT, OTHER_BOND)].to_numpy(dtype="float64")).all()


def test_nonpositive_terminal_stock_input_fails_closed_without_infinity() -> None:
    sources = _daily_sources()
    base = sources["market_cbond.daily_base"]
    row = base.index[(base["code"] == "110001") & (base["trade_date"] == base["trade_date"].max())][0]
    base.loc[row, "stk_prev_close_price"] = 0.0

    frame = build_factor_frame(_panel(), _specs(), daily_data=sources)

    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame.loc[(DT, BOND)].isna().all()


def test_duplicate_normalized_daily_bond_key_is_rejected() -> None:
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


def test_inconsistent_shared_underlying_return_is_rejected() -> None:
    sources = _daily_sources()
    base = sources["market_cbond.daily_base"]
    last_day = base["trade_date"].max()
    row = base.index[(base["code"] == "110005") & (base["trade_date"] == last_day)][0]
    base.loc[row, "stk_close_price"] = float(base.loc[row, "stk_close_price"]) * 1.1

    with pytest.raises(ValueError, match="inconsistent strict-prior underlying return"):
        build_factor_frame(_panel(), _specs(), daily_data=sources)
