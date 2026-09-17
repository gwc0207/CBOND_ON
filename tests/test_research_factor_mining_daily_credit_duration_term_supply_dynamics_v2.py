from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_credit_duration_term_supply_dynamics_v2 as dynamics,
)
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(1, 9))
_CORRELATION_SIGNALS = (
    *dynamics._CREDIT_MOTION_SIGNALS,
    "tsrr2_supply_term_level_corr20",
    "cfar_yield_amount_capacity_delta_corr20",
    "cfar_yield_volume_capacity_delta_corr20",
    "catr_convexity_duration_delta_corr20",
    "catr_convexity_anchor_delta_corr20",
)


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
            close = (
                98.0 + 1.4 * code_index + 0.14 * position + 1.1 * np.sin(phase * 0.23)
            )
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
            conv_value = (
                91.0 + 0.52 * code_index + 0.19 * position + 1.4 * np.sin(phase * 0.25)
            )
            redemption = (
                84.0
                + 0.17 * code_index
                + 0.045 * position
                + 0.35 * np.cos(phase * 0.16)
            )
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
        FactorSpec(
            name=entry.signal,
            factor=dynamics.KERNEL_NAME,
            params={"signal": entry.signal},
        )
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


def test_catalogue_has_twelve_distinct_robust_signals_and_explicit_context() -> None:
    entries = dynamics.factor_mining_catalog()
    requirements = (
        dynamics.FactorMiningDailyCreditDurationTermSupplyDynamicsV2.daily_requirements(
            {"signal": "catr_convexity_state_standardized_oos20"}
        )
    )
    context_requirements = infer_factor_context_requirements(_specs())

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "prior_credit_duration_convexity_motion_robust": 3,
        "prior_term_supply_roll_rebalancing_robust": 3,
        "prior_credit_flow_absorption_dynamics_robust": 3,
        "prior_convexity_anchor_transition_robust": 3,
    }
    assert set(dynamics.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(dynamics.KERNEL_NAME)
        is dynamics.FactorMiningDailyCreditDurationTermSupplyDynamicsV2
    )
    assert [item.source for item in requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_base",
    ]
    assert requirements[0].columns == (
        "exchange_code",
        "close_price",
        "volume",
        "amount",
    )
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
    assert context_requirements.daily_required is True
    assert context_requirements.stock_panel_required is False
    assert context_requirements.bond_stock_map_required is False
    assert dynamics._MIN_EFFECTIVE_SCALE > 0.0
    assert dynamics._MAX_STANDARDIZED_CONDITION_NUMBER < np.inf


def test_twelve_signals_build_without_inf_and_correlations_are_bounded() -> None:
    frame = _build()

    assert frame.columns.tolist() == [
        entry.signal for entry in dynamics.factor_mining_catalog()
    ]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()
    assert (frame.nunique(dropna=True) > 1).all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    correlations = frame.loc[:, list(_CORRELATION_SIGNALS)].to_numpy(dtype="float64")
    assert (np.abs(correlations) <= 1.0 + 1e-12).all()


def test_low_scale_and_ill_conditioned_regressions_fail_closed() -> None:
    tiny = np.linspace(0.0, dynamics._MIN_EFFECTIVE_SCALE / 10.0, dynamics._WINDOW)
    variable = np.linspace(-1.0, 1.0, dynamics._WINDOW)
    assert np.isnan(dynamics._safe_corr(tiny, variable))

    axis = np.linspace(-1.0, 1.0, dynamics._WINDOW)
    training = np.column_stack(
        (
            np.sin(axis),
            axis,
            axis * (1.0 + 1e-10),
        )
    )
    current = np.array([0.1, 1.1, 1.1 * (1.0 + 1e-10)], dtype="float64")
    assert np.isnan(dynamics._one_standardized_oos_residual(training, current))


def test_score_day_future_stale_and_gap_histories_fail_closed() -> None:
    baseline = _build()
    contaminated = _build(_contaminate(_daily_sources()))
    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)

    sources = _daily_sources()
    latest = sources["market_cbond.daily_price"]["trade_date"].max()
    stale_code = _bare(CODES[0])
    stale_base = (
        sources["market_cbond.daily_base"]
        .loc[
            ~(
                (sources["market_cbond.daily_base"]["code"] == stale_code)
                & (sources["market_cbond.daily_base"]["trade_date"] == latest)
            )
        ]
        .copy()
    )
    stale = _build(
        {
            "market_cbond.daily_price": sources["market_cbond.daily_price"],
            "market_cbond.daily_base": stale_base,
        }
    )
    assert stale.loc[(SCORE, CODES[0])].isna().all()

    gap_day = sorted(sources["market_cbond.daily_price"]["trade_date"].unique())[-2]
    gap_base = (
        sources["market_cbond.daily_base"]
        .loc[
            ~(
                (sources["market_cbond.daily_base"]["code"] == _bare(CODES[1]))
                & (sources["market_cbond.daily_base"]["trade_date"] == gap_day)
            )
        ]
        .copy()
    )
    gap = _build(
        {
            "market_cbond.daily_price": sources["market_cbond.daily_price"],
            "market_cbond.daily_base": gap_base,
        }
    )
    assert gap.loc[(SCORE, CODES[1])].isna().all()
    assert not np.isinf(stale.to_numpy(dtype="float64")).any()
    assert not np.isinf(gap.to_numpy(dtype="float64")).any()


def test_missing_field_duplicate_key_bad_anchor_and_unknown_signal_are_rejected() -> (
    None
):
    kernel = dynamics.FactorMiningDailyCreditDurationTermSupplyDynamicsV2()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={},
                params={"signal": "catr_convexity_state_standardized_oos20"},
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
                params={"signal": "catr_convexity_state_standardized_oos20"},
            )
        )

    duplicate_price = pd.concat(
        [
            sources["market_cbond.daily_price"],
            sources["market_cbond.daily_price"].iloc[[0]],
        ],
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
                params={"signal": "catr_convexity_state_standardized_oos20"},
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
    assert (
        frame.loc[(SCORE, CODES[0]), list(dynamics._CONVEXITY_ANCHOR_SIGNALS)[1:]]
        .isna()
        .all()
    )

    with pytest.raises(KeyError, match="unknown signal"):
        kernel.compute(
            FactorComputeContext(panel=_panel(), params={"signal": "unknown"})
        )
