from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_asymmetric_state_transitions_v1 as transitions,
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
    dates = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=75)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate((BOND, OTHER_BOND)):
        bare, exchange = code.split(".")
        prior_close = 100.0 + 3.0 * code_index
        for position, day in enumerate(dates):
            gap = 0.003 * np.sin(0.43 * position + 0.29 * code_index)
            body = 0.0025 * np.cos(0.31 * position + 0.41 * code_index)
            open_price = prior_close * np.exp(gap)
            close_price = open_price * np.exp(body)
            high_price = max(open_price, close_price) * np.exp(
                0.002 + 0.0008 * (1.0 + np.sin(0.22 * position))
            )
            low_price = min(open_price, close_price) * np.exp(
                -0.002 - 0.0007 * (1.0 + np.cos(0.27 * position))
            )
            price_rows.append(
                {
                    "trade_date": day,
                    "code": bare,
                    "exchange_code": exchange,
                    "act_prev_close_price": prior_close,
                    "open_price": open_price,
                    "high_price": high_price,
                    "low_price": low_price,
                    "close_price": close_price,
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
                        "current_yield": 0.014
                        + 0.0009 * np.sin(0.37 * position + 0.23 * code_index),
                        "duration": 2.9
                        - 0.004 * position
                        + 0.045 * np.sin(0.19 * position + code_index),
                        "convexity": 1.6
                        + 0.08 * np.cos(0.31 * position + 0.17 * code_index),
                        "stock_volatility": 0.28
                        + 0.045 * np.sin(0.37 * position + 0.53 * code_index),
                    }
                )
            prior_close = close_price
    if include_score_and_future:
        for day in (DT.normalize(), DT.normalize() + pd.Timedelta(days=1)):
            for bare, exchange in (("110001", "SH"), ("110002", "SH")):
                price_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "act_prev_close_price": 1.0,
                        "open_price": 9_999_999.0,
                        "high_price": 99_999_999.0,
                        "low_price": 1.0,
                        "close_price": 9_999_999.0,
                    }
                )
                base_rows.append(
                    {
                        "trade_date": day,
                        "code": bare,
                        "exchange_code": exchange,
                        "current_yield": 9_999.0,
                        "duration": 1.0,
                        "convexity": 9_999.0,
                        "stock_volatility": 9_999.0,
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
            factor=transitions.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in transitions.factor_mining_catalog()
    ]


def _build(**source_kwargs: object) -> pd.DataFrame:
    return build_factor_frame(
        _panel(), _specs(), daily_data=_daily_sources(**source_kwargs)
    )


def test_catalogue_has_three_asymmetric_state_transition_families() -> None:
    entries = transitions.factor_mining_catalog()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "prior_yield_directional_price_pass_through": 3,
        "prior_duration_convexity_range_transition": 3,
        "prior_stockvol_candle_reversal_state": 3,
    }
    assert set(transitions.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(transitions.KERNEL_NAME)
        is transitions.FactorMiningDailyAsymmetricStateTransitionsV1
    )
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in transitions.FactorMiningDailyAsymmetricStateTransitionsV1.daily_requirements()
    ] == [
        (
            "market_cbond.daily_price",
            (
                "exchange_code",
                "act_prev_close_price",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
            ),
            75,
        ),
        (
            "market_cbond.daily_base",
            (
                "exchange_code",
                "current_yield",
                "duration",
                "convexity",
                "stock_volatility",
            ),
            75,
        ),
    ]


def test_all_nine_state_transition_signals_are_finite_with_strict_prior_histories() -> (
    None
):
    frame = _build()

    assert frame.columns.tolist() == [
        entry.signal for entry in transitions.factor_mining_catalog()
    ]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()


def test_score_day_and_future_rows_cannot_change_strict_prior_transition_outputs() -> (
    None
):
    baseline = _build()
    contaminated = _build(include_score_and_future=True)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_tminus1_base_row_fails_closed_for_that_security() -> None:
    frame = _build(stale_base_for_bond=True)

    assert frame.loc[(DT, BOND)].isna().all()
    assert np.isfinite(frame.loc[(DT, OTHER_BOND)].to_numpy(dtype="float64")).all()


def test_nonpositive_stock_volatility_fails_closed_without_infinity() -> None:
    sources = _daily_sources()
    base = sources["market_cbond.daily_base"]
    last_bond_row = base.index[
        (base["code"] == "110001") & (base["trade_date"] == base["trade_date"].max())
    ][0]
    base.loc[last_bond_row, "stock_volatility"] = 0.0

    frame = build_factor_frame(_panel(), _specs(), daily_data=sources)

    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame.loc[(DT, BOND), list(transitions._STOCK_VOL_SIGNALS)].isna().all()


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
