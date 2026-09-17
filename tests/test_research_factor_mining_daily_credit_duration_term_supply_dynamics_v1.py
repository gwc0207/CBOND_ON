from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_credit_duration_term_supply_dynamics_v1 as dynamics,
)
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(1, 9))


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _panel() -> pd.DataFrame:
    rows = [
        {"dt": SCORE, "code": code, "seq": sequence, "trade_time": SCORE, "last": 100.0}
        for code in CODES
        for sequence in range(2)
    ]
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=66)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        supply_base = 850_000_000.0 + 65_000_000.0 * code_index
        for position, day in enumerate(days):
            phase = float(position + 1.7 * code_index)
            close = 98.0 + 1.4 * code_index + 0.14 * position + 1.1 * np.sin(phase * 0.23)
            volume = 82_000.0 * (1.0 + 0.035 * code_index + 0.18 * np.cos(phase * 0.29))
            amount = volume * close * (1.0 + 0.025 * np.sin(phase * 0.17 + 0.4))
            supply_event = 1.0 - 0.017 * float((position + 2 * code_index) % 13 == 0)
            supply = supply_base * (1.0 - 0.00075 * position) * supply_event
            current_yield = (
                0.012
                + 0.00032 * code_index
                + 0.000021 * position
                + 0.00046 * np.sin(phase * 0.31)
                + 0.00012 * np.cos(phase * 0.11)
            )
            duration = (
                3.65
                + 0.09 * code_index
                - 0.008 * position
                + 0.075 * np.sin(phase * 0.19 + 0.2)
            )
            convexity = (
                8.8
                + 0.36 * code_index
                + 0.018 * position
                + 0.23 * np.cos(phase * 0.27)
                + 0.04 * current_yield * 10_000.0
            )
            year_to_mat = 4.8 + 0.12 * code_index - 0.0040 * position
            premium = 0.15 + 0.009 * code_index + 0.012 * np.sin(phase * 0.21)
            conv_value = 91.0 + 0.52 * code_index + 0.19 * position + 1.4 * np.sin(phase * 0.25)
            redemption = 84.0 + 0.17 * code_index + 0.045 * position + 0.35 * np.cos(phase * 0.16)
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": "SH",
                    "close_price": close,
                    "volume": volume,
                    "amount": amount,
                }
            )
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": "SH",
                    "current_yield": current_yield,
                    "duration": duration,
                    "convexity": convexity,
                    "year_to_mat": year_to_mat,
                    "remain_size": supply,
                    "bond_prem_ratio": premium,
                    "conv_value": conv_value,
                    "pure_redemption_value": redemption,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=dynamics.KERNEL_NAME, params={"signal": entry.signal})
        for entry in dynamics.factor_mining_catalog()
    ]


def _build(sources: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    return build_factor_frame(
        _panel(),
        _specs(),
        daily_data=_daily_sources() if sources is None else sources,
    )


def _contaminate(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        score_day = terminal.copy()
        future = terminal.copy()
        score_day["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column not in {"trade_date", "code", "exchange_code"}:
                score_day[column] = 1_000_000.0
                future[column] = 2_000_000.0
        out[source] = pd.concat([frame, score_day, future], ignore_index=True)
    return out


def test_catalogue_is_twelve_signals_across_four_distinct_dynamic_families() -> None:
    entries = dynamics.factor_mining_catalog()
    requirements = dynamics.FactorMiningDailyCreditDurationTermSupplyDynamicsV1.daily_requirements(
        {"signal": "cdcm_yield_duration_delta_beta20"}
    )
    context_requirements = infer_factor_context_requirements(_specs())

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "prior_credit_duration_convexity_motion": 3,
        "prior_term_supply_roll_rebalancing": 3,
        "prior_credit_flow_absorption_dynamics": 3,
        "prior_convexity_anchor_transition": 3,
    }
    assert set(dynamics.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(dynamics.KERNEL_NAME) is dynamics.FactorMiningDailyCreditDurationTermSupplyDynamicsV1
    assert [item.source for item in requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_base",
    ]
    assert requirements[0].columns == ("exchange_code", "close_price", "volume", "amount")
    assert requirements[1].columns == (
        "exchange_code",
        "current_yield",
        "duration",
        "convexity",
        "year_to_mat",
        "remain_size",
        "bond_prem_ratio",
        "conv_value",
        "pure_redemption_value",
    )
    assert all(item.lookback_days >= 66 for item in requirements)
    assert context_requirements.daily_required is True
    assert context_requirements.stock_panel_required is False
    assert context_requirements.bond_stock_map_required is False
    assert "debt_puredebt_ratio" not in dynamics._BASE_FIELDS
    assert "puredebt_prem_ratio" not in dynamics._BASE_FIELDS


def test_all_twelve_signals_build_finite_from_complete_strict_prior_histories() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in dynamics.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()
    assert (frame.nunique(dropna=True) > 1).all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_score_day_and_future_rows_cannot_change_tminus1_outputs() -> None:
    baseline = _build()
    contaminated = _build(_contaminate(_daily_sources()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_anchor_and_nonconsecutive_base_history_fail_closed() -> None:
    sources = _daily_sources()
    latest = sources["market_cbond.daily_price"]["trade_date"].max()
    stale_code = _bare(CODES[0])
    stale_base = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == stale_code)
            & (sources["market_cbond.daily_base"]["trade_date"] == latest)
        )
    ].copy()
    stale = _build(
        {
            "market_cbond.daily_price": sources["market_cbond.daily_price"],
            "market_cbond.daily_base": stale_base,
        }
    )

    gap_day = sorted(sources["market_cbond.daily_price"]["trade_date"].unique())[-2]
    gap_base = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == _bare(CODES[1]))
            & (sources["market_cbond.daily_base"]["trade_date"] == gap_day)
        )
    ].copy()
    gap = _build(
        {
            "market_cbond.daily_price": sources["market_cbond.daily_price"],
            "market_cbond.daily_base": gap_base,
        }
    )

    assert stale.loc[(SCORE, CODES[0])].isna().all()
    assert gap.loc[(SCORE, CODES[1])].isna().all()
    assert not np.isinf(stale.to_numpy(dtype="float64")).any()
    assert not np.isinf(gap.to_numpy(dtype="float64")).any()


def test_missing_field_duplicate_prior_row_and_bad_anchor_fail_closed() -> None:
    kernel = dynamics.FactorMiningDailyCreditDurationTermSupplyDynamicsV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={},
                params={"signal": "cdcm_yield_duration_delta_beta20"},
            )
        )

    sources = _daily_sources()
    missing = sources["market_cbond.daily_base"].drop(columns=["convexity"])
    with pytest.raises(KeyError, match="convexity"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={
                    "market_cbond.daily_price": sources["market_cbond.daily_price"],
                    "market_cbond.daily_base": missing,
                },
                params={"signal": "cdcm_yield_duration_delta_beta20"},
            )
        )

    duplicate_price = pd.concat(
        [sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={
                    "market_cbond.daily_price": duplicate_price,
                    "market_cbond.daily_base": sources["market_cbond.daily_base"],
                },
                params={"signal": "cdcm_yield_duration_delta_beta20"},
            )
        )

    broken = _daily_sources()
    latest = broken["market_cbond.daily_price"]["trade_date"].max()
    latest_row = broken["market_cbond.daily_base"].index[
        (broken["market_cbond.daily_base"]["code"] == _bare(CODES[0]))
        & (broken["market_cbond.daily_base"]["trade_date"] == latest)
    ][0]
    broken["market_cbond.daily_base"].loc[latest_row, "pure_redemption_value"] = 0.0
    frame = _build(broken)
    assert frame.loc[(SCORE, CODES[0]), list(dynamics._CONVEXITY_ANCHOR_SIGNALS)[1:]].isna().all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_unknown_signal_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        dynamics.FactorMiningDailyCreditDurationTermSupplyDynamicsV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "unknown"})
        )
