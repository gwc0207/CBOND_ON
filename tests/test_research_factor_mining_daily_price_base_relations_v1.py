from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_price_base_relations_v1 as relations,
)
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
OTHER_BOND = "110002.SH"


def _panel() -> pd.DataFrame:
    rows = [
        {"dt": DT, "code": BOND, "seq": 0, "last": 101.0},
        {"dt": DT, "code": OTHER_BOND, "seq": 0, "last": 102.0},
    ]
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _daily_sources(
    *,
    include_score_and_future: bool = False,
    stale_base_for_bond: bool = False,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=45)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate((BOND, OTHER_BOND)):
        bare, exchange = code.split(".")
        for position, day in enumerate(dates):
            close = (
                100.0
                + 3.0 * code_index
                + 0.13 * position
                + 0.6 * np.sin(position * (0.37 + 0.03 * code_index))
            )
            volume = (
                4_000.0
                + 100.0 * code_index
                + 19.0 * position
                + 70.0 * np.cos(position * 0.21)
            )
            amount = (
                volume * close * (1.0 + 0.0007 * np.sin(position * 0.31 + code_index))
            )
            price_rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "close_price": close,
                    "volume": volume,
                    "amount": amount,
                }
            )
            if not (
                stale_base_for_bond and code == BOND and position == len(dates) - 1
            ):
                base_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "remain_size": 180_000_000.0
                        - 17_000.0 * position
                        - 1_000_000.0 * code_index,
                        "bond_prem_ratio": 0.18
                        + 0.003 * np.sin(position * 0.29 + code_index),
                        "current_yield": 0.014
                        + 0.0002 * np.cos(position * 0.19 + code_index),
                        "conv_value": 91.0
                        + 0.17 * position
                        + 0.9 * code_index
                        + 0.3 * np.sin(position * 0.23),
                        "pure_redemption_value": 84.0
                        + 0.05 * position
                        + 0.2 * code_index
                        + 0.08 * np.cos(position * 0.27),
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
                        "close_price": 9_999_999.0,
                        "volume": 9_999_999.0,
                        "amount": 9_999_999_999.0,
                    }
                )
                base_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "remain_size": 1.0,
                        "bond_prem_ratio": 9_999.0,
                        "current_yield": 9_999.0,
                        "conv_value": 9_999_999.0,
                        "pure_redemption_value": 1.0,
                    }
                )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=relations.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in relations.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(
        _panel(), _specs(), daily_data=_daily_sources(**source_kwargs)
    )


def test_catalogue_has_three_distinct_relation_families_and_registered_kernel() -> None:
    entries = relations.factor_mining_catalog()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "prior_structural_capacity_utilization": 3,
        "prior_valuation_flow_elasticity": 3,
        "prior_intrinsic_anchor_topology": 3,
    }
    assert set(relations.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(relations.KERNEL_NAME)
        is relations.FactorMiningDailyPriceBaseRelationsV1
    )
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in relations.FactorMiningDailyPriceBaseRelationsV1.daily_requirements()
    ] == [
        (
            "market_cbond.daily_price",
            ("exchange_code", "close_price", "volume", "amount"),
            45,
        ),
        (
            "market_cbond.daily_base",
            (
                "exchange_code",
                "remain_size",
                "bond_prem_ratio",
                "current_yield",
                "conv_value",
                "pure_redemption_value",
            ),
            45,
        ),
    ]


def test_all_nine_relation_signals_are_finite_with_strict_prior_histories() -> None:
    frame = _build()

    assert frame.columns.tolist() == [
        entry.signal for entry in relations.factor_mining_catalog()
    ]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()


def test_score_day_and_future_daily_rows_cannot_change_strict_prior_outputs() -> None:
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_tminus1_base_row_fails_closed_for_that_security() -> None:
    frame = _build(stale_base_for_bond=True)

    assert frame.loc[(DT, BOND)].isna().all()
    assert np.isfinite(frame.loc[(DT, OTHER_BOND)].to_numpy(dtype="float64")).all()


def test_nonpositive_intrinsic_anchor_fails_closed_without_infinity() -> None:
    sources = _daily_sources()
    base = sources["market_cbond.daily_base"]
    last_bond_row = base.index[
        (base["code"] == "110001") & (base["trade_date"] == base["trade_date"].max())
    ][0]
    base.loc[last_bond_row, "pure_redemption_value"] = 0.0

    frame = build_factor_frame(_panel(), _specs(), daily_data=sources)

    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame.loc[(DT, BOND), list(relations._ANCHOR_SIGNALS)].isna().all()


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
