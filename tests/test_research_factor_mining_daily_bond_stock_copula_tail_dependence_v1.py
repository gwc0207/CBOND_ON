from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_bond_stock_copula_tail_dependence_v1 as copula_tail,
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
    stale_base_for_bond: bool = False,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=72)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate((BOND, OTHER_BOND)):
        bare, exchange = code.split(".")
        bond_price = 100.0 + 3.0 * code_index
        stock_price = 15.0 + 2.0 * code_index
        for position, day in enumerate(dates):
            stock_return = 0.016 * np.sin(0.43 * position + 0.51 * code_index)
            bond_return = (
                0.55 * stock_return
                + 0.007 * np.sin(0.91 * position + 0.19 * code_index)
                + 0.004 * float(stock_return > 0.0)
            )
            stock_previous = stock_price
            bond_previous = bond_price
            stock_price *= float(np.exp(stock_return))
            bond_price *= float(np.exp(bond_return))
            price_rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "prev_close_price": bond_previous,
                    "close_price": bond_price,
                }
            )
            if not (stale_base_for_bond and code == BOND and position == len(dates) - 1):
                base_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "stk_prev_close_price": stock_previous,
                        "stk_close_price": stock_price,
                    }
                )
    if include_score_and_future:
        for day in (DT.normalize(), DT.normalize() + pd.Timedelta(days=1)):
            for bare, exchange in (("110001", "SH"), ("110002", "SH")):
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
        FactorSpec(name=entry.signal, factor=copula_tail.KERNEL_NAME, params={"signal": entry.signal})
        for entry in copula_tail.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(**source_kwargs))


def test_catalogue_daily_requirements_and_registry_are_explicit() -> None:
    entries = copula_tail.factor_mining_catalog()

    assert len(entries) == 3
    assert Counter(entry.family for entry in entries) == {
        "prior_bond_stock_copula_tail_dependence": 3
    }
    assert set(copula_tail.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(copula_tail.KERNEL_NAME)
        is copula_tail.FactorMiningDailyBondStockCopulaTailDependenceV1
    )
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in copula_tail.FactorMiningDailyBondStockCopulaTailDependenceV1.daily_requirements()
    ] == [
        ("market_cbond.daily_price", ("exchange_code", "prev_close_price", "close_price"), 75),
        (
            "market_cbond.daily_base",
            ("exchange_code", "stk_prev_close_price", "stk_close_price"),
            75,
        ),
    ]


def test_strict_prior_tail_signals_are_finite_and_algebraically_consistent() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in copula_tail.factor_mining_catalog()]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()
    np.testing.assert_allclose(
        frame["bsct_tail_dependence_asymmetry60"].to_numpy(dtype="float64"),
        frame["bsct_upper_tail_dependence60"].to_numpy(dtype="float64")
        - frame["bsct_lower_tail_dependence60"].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=0.0,
    )


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
