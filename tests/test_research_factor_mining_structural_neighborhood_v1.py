from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_structural_neighborhood_v1 as structural
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
_RATINGS = ("AAA", "AA+", "AA", "AA-")
_GROUP_SIZE = 4
CODES = tuple(
    f"{110001 + index:06d}.{'SH' if index % 2 == 0 else 'SZ'}"
    for index in range(len(_RATINGS) * 2 * _GROUP_SIZE)
)


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _panel() -> pd.DataFrame:
    rows = [{"dt": SCORE, "code": code, "seq": seq} for code in CODES for seq in range(2)]
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=70)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        cohort_index = code_index // _GROUP_SIZE
        member = code_index % _GROUP_SIZE
        rating_index = cohort_index // 2
        term_index = cohort_index % 2
        for day_index, day in enumerate(days):
            phase = 0.71 * day_index + 1.37 * code_index
            close = 96.0 + 0.68 * code_index + 0.14 * day_index + 1.1 * np.sin(phase / 3.0)
            prev_close = close * (1.0 - 0.0035 * np.sin((phase - 0.4) / 2.2))
            amount = (
                18_000_000.0
                + 370_000.0 * code_index
                + 31_000.0 * day_index
                + 1_900_000.0 * (1.0 + np.sin(phase / 2.7))
            )
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "prev_close_price": prev_close,
                    "close_price": close,
                    "amount": amount,
                }
            )

            maturity_base = 1.25 if term_index == 0 else 3.25
            maturity = maturity_base + 0.055 * member - 0.001 * day_index
            duration = 0.69 * maturity + 0.045 * member + 0.014 * np.cos(phase / 4.0)
            premium = 0.075 + 0.014 * rating_index + 0.003 * member + 0.00022 * day_index + 0.004 * np.sin(phase)
            ytm = 0.010 + 0.0016 * rating_index + 0.00003 * day_index + 0.0008 * np.cos(phase / 2.4)
            remain_size = 900_000_000.0 + 95_000_000.0 * code_index - 700_000.0 * day_index
            stock_close = 8.0 + 0.18 * code_index + 0.027 * day_index + 0.38 * np.sin(phase / 2.1)
            stock_volatility = 0.16 + 0.006 * rating_index + 0.0025 * member + 0.004 * (1.0 + np.cos(phase / 5.0))
            stock_amount = (
                650_000_000.0
                + 21_000_000.0 * code_index
                + 2_400_000.0 * day_index
                + 70_000_000.0 * (1.0 + np.cos(phase / 2.9))
            )
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "year_to_mat": maturity,
                    "duration": duration,
                    "bond_prem_ratio": premium,
                    "ytm": ytm,
                    "remain_size": remain_size,
                    "rating": _RATINGS[rating_index],
                    "stock_code": f"{600001 + code_index:06d}.SH",
                    "stock_close_price": stock_close,
                    "stock_volatility": stock_volatility,
                    "stk_amount": stock_amount,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=structural.KERNEL_NAME, params={"signal": entry.signal})
        for entry in structural.structural_neighborhood_catalog()
    ]


def _contaminate_with_score_day_and_future_rows(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        score_day = terminal.copy()
        future = terminal.copy()
        score_day["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column in {"trade_date", "code", "exchange_code", "rating", "stock_code"}:
                continue
            score_day[column] = 1_000_000.0
            future[column] = 2_000_000.0
        out[source] = pd.concat([frame, score_day, future], ignore_index=True)
    return out


def test_catalogue_has_four_distinct_families_and_twelve_signals() -> None:
    entries = structural.structural_neighborhood_catalog()

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "structural_peer_state_transmission": 3,
        "structural_neighborhood_geometry": 3,
        "rating_term_cohort_deviation": 3,
        "underlying_stock_state_cohort": 3,
    }
    assert {entry.kernel for entry in entries} == {structural.KERNEL_NAME}
    assert FactorRegistry.get(structural.KERNEL_NAME) is structural.FactorMiningStructuralNeighborhoodV1
    assert structural.FactorMiningStructuralNeighborhoodV1.__name__ not in factor_defs.__all__
    assert structural.FactorMiningStructuralNeighborhoodV1.requires_stock_panel is False
    assert structural.FactorMiningStructuralNeighborhoodV1.requires_bond_stock_map is False


def test_requirements_are_family_specific_and_use_only_daily_price_and_daily_base() -> None:
    peer_requirements = structural.FactorMiningStructuralNeighborhoodV1.daily_requirements(
        {"signal": "spt_peer_return1_mean"}
    )
    assert [item.source for item in peer_requirements] == ["market_cbond.daily_price", "market_cbond.daily_base"]
    assert peer_requirements[0].columns == ("exchange_code", "prev_close_price", "close_price")
    assert peer_requirements[1].columns == (
        "exchange_code",
        "year_to_mat",
        "duration",
        "bond_prem_ratio",
        "ytm",
        "remain_size",
    )

    stock_requirements = structural.FactorMiningStructuralNeighborhoodV1.daily_requirements(
        {"signal": "smc_peer_stock_return1_mean"}
    )
    assert "stock_code" in stock_requirements[1].columns
    assert "stock_close_price" in stock_requirements[1].columns
    assert "stk_amount" in stock_requirements[1].columns
    assert all(item.lookback_days >= 65 for item in stock_requirements)

    all_requirements = structural.FactorMiningStructuralNeighborhoodV1.daily_requirements()
    assert {item.source for item in all_requirements} == {"market_cbond.daily_price", "market_cbond.daily_base"}
    assert all("exchange_code" in item.columns for item in all_requirements)


def test_all_signals_build_with_cross_sectional_support_and_without_inf() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(), workers=4)

    assert frame.columns.tolist() == [entry.signal for entry in structural.structural_neighborhood_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= len(CODES) - 2).all()


def test_all_signals_ignore_score_day_and_future_daily_outliers() -> None:
    baseline = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(), workers=4)
    contaminated = build_factor_frame(
        _panel(),
        _specs(),
        daily_data=_contaminate_with_score_day_and_future_rows(_daily_sources()),
        workers=4,
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_source_or_field_and_duplicate_history_fail_closed() -> None:
    kernel = structural.FactorMiningStructuralNeighborhoodV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(FactorComputeContext(panel=_panel(), daily_data={}, params={"signal": "spt_peer_return1_mean"}))

    sources = _daily_sources()
    missing = sources["market_cbond.daily_base"].drop(columns=["year_to_mat"])
    with pytest.raises(KeyError, match="year_to_mat"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": missing},
                params={"signal": "spt_peer_return1_mean"},
            )
        )

    duplicate_price = pd.concat(
        [sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": duplicate_price, "market_cbond.daily_base": sources["market_cbond.daily_base"]},
                params={"signal": "spt_peer_return1_mean"},
            )
        )


def test_missing_or_mismatched_base_anchor_is_nan_not_carried_forward() -> None:
    sources = _daily_sources()
    target = CODES[0]
    anchor = sources["market_cbond.daily_price"]["trade_date"].max()
    stale_base = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == _bare(target))
            & (sources["market_cbond.daily_base"]["exchange_code"] == _exchange(target))
            & (sources["market_cbond.daily_base"]["trade_date"] == anchor)
        )
    ].copy()
    frame = build_factor_frame(
        _panel(),
        _specs(),
        daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": stale_base},
        workers=4,
    )

    assert frame.loc[(SCORE, target)].isna().all()
    assert frame.loc[(SCORE, CODES[-1])].notna().any()


def test_zero_stock_amount_is_nan_wedge_not_division_error() -> None:
    sources = _daily_sources()
    anchor = sources["market_cbond.daily_price"]["trade_date"].max()
    target = CODES[0]
    base = sources["market_cbond.daily_base"].copy()
    base.loc[
        (base["trade_date"] == anchor)
        & (base["code"] == _bare(target))
        & (base["exchange_code"] == _exchange(target)),
        "stk_amount",
    ] = 0.0

    frame = build_factor_frame(
        _panel(),
        _specs(),
        daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": base},
        workers=1,
    )

    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_local_peer_outputs_exclude_self_and_keep_rating_term_cohorts_conditional() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(), workers=1)
    target = CODES[0]
    same_cohort = CODES[1:4]
    sources = _daily_sources()
    latest = sources["market_cbond.daily_price"].loc[
        sources["market_cbond.daily_price"]["trade_date"] == sources["market_cbond.daily_price"]["trade_date"].max()
    ].copy()
    returns = {
        f"{row.code}.{row.exchange_code}": row.close_price / row.prev_close_price - 1.0
        for row in latest.itertuples(index=False)
    }

    value = float(frame.loc[(SCORE, target), "spt_peer_return1_mean"])
    assert np.isfinite(value)
    assert value != pytest.approx(returns[target])
    assert np.isfinite(float(frame.loc[(SCORE, target), "rtc_premium_gap_median"]))
    assert len(same_cohort) == 3
