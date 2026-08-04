from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_daily_mark_barrier_dynamics_v1 as dynamics
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(8)) + tuple(
    f"127{index:03d}.SZ" for index in range(8)
)


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _panel() -> pd.DataFrame:
    rows = [{"dt": SCORE, "code": code, "seq": seq} for code in CODES for seq in range(2)]
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=66)
    price_rows: list[dict[str, object]] = []
    twap_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        for day_index, day in enumerate(days):
            phase = float(day_index + 1.9 * code_index)
            reference = 100.0 + 0.8 * code_index + 0.06 * day_index
            lunch_end = reference * (1.0 + 0.0012 * np.sin(phase / 3.7))
            reopen = lunch_end * (1.0 + 0.0018 * np.cos(phase / 4.1 + 0.2 * code_index))
            afternoon_end = reopen * (1.0 + 0.0014 * np.sin(phase / 5.3))
            late = afternoon_end * (1.0 + 0.0017 * np.cos(phase / 3.1 + 0.1 * code_index))
            execution = late * (1.0 + 0.0015 * np.sin(phase / 2.9 + 0.3 * code_index))
            close = execution * (1.0 + 0.0011 * np.sin(phase / 2.2) + 0.0005 * np.cos(phase / 4.6))
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "close_price": close,
                }
            )
            twap_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "twap_1100_1130": lunch_end,
                    "twap_1300_1330": reopen,
                    "twap_1400_1430": afternoon_end,
                    "twap_1430_1442": late,
                    "twap_1442_1457": execution,
                }
            )

            stock = 10.0 + 0.08 * code_index + 3.1 * np.sin(0.37 * day_index + 0.47 * code_index)
            put = 7.0 + 0.04 * code_index + 0.025 * np.sin(phase / 8.0)
            call = 13.0 + 0.04 * code_index + 0.025 * np.cos(phase / 7.0)
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "cb_put_price": put,
                    "cb_call_price": call,
                    "stock_close_price": stock,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_twap": pd.DataFrame(twap_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=dynamics.KERNEL_NAME, params={"signal": entry.signal})
        for entry in dynamics.factor_mining_catalog()
    ]


def _build(sources: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources() if sources is None else sources)


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


def test_catalogue_has_three_research_only_families_and_exact_context_contracts() -> None:
    entries = dynamics.factor_mining_catalog()
    all_requirements = dynamics.FactorMiningDailyMarkBarrierDynamicsV1.daily_requirements()
    final_requirements = dynamics.FactorMiningDailyMarkBarrierDynamicsV1.daily_requirements(
        {"signal": "fmed_mark_execution_log_basis"}
    )
    barrier_requirements = dynamics.FactorMiningDailyMarkBarrierDynamicsV1.daily_requirements(
        {"signal": "boh_edge_state_flip_rate20"}
    )

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "final_mark_execution_dislocation": 3,
        "phase_conditioned_finalization": 3,
        "barrier_occupancy_hysteresis": 3,
    }
    assert set(dynamics.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(dynamics.KERNEL_NAME) is dynamics.FactorMiningDailyMarkBarrierDynamicsV1
    assert dynamics.FactorMiningDailyMarkBarrierDynamicsV1.__name__ not in factor_defs.__all__

    assert [item.source for item in final_requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_twap",
    ]
    assert final_requirements[0].columns == ("exchange_code", "close_price")
    assert final_requirements[1].columns == (
        "exchange_code",
        "twap_1430_1442",
        "twap_1442_1457",
    )
    assert [item.source for item in barrier_requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_base",
    ]
    assert barrier_requirements[1].columns == (
        "exchange_code",
        "cb_put_price",
        "cb_call_price",
        "stock_close_price",
    )
    assert {item.source for item in all_requirements} == {
        "market_cbond.daily_price",
        "market_cbond.daily_twap",
        "market_cbond.daily_base",
    }
    assert all(item.lookback_days >= 66 for item in all_requirements)
    context = infer_factor_context_requirements(_specs())
    assert context.daily_required is True
    assert context.stock_panel_required is False
    assert context.bond_stock_map_required is False


def test_all_nine_signals_build_without_inf_and_have_cross_sectional_support() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in dynamics.factor_mining_catalog()]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert (frame.notna().sum(axis=0) == len(CODES)).all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame["fmed_mark_execution_log_basis"].abs().sum() > 0.0
    assert frame["pcf_final_mark_oos_residual40"].abs().sum() > 0.0
    assert frame["boh_edge_state_flip_rate20"].sum() > 0.0


def test_score_day_and_future_daily_rows_are_ignored_exactly() -> None:
    baseline = _build()
    contaminated = _build(_contaminate(_daily_sources()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_anchor_and_interior_gaps_fail_closed_for_the_affected_family() -> None:
    sources = _daily_sources()
    anchor = sources["market_cbond.daily_price"]["trade_date"].max()

    stale_twap_code = _bare(CODES[0])
    stale_twap = sources["market_cbond.daily_twap"].loc[
        ~(
            (sources["market_cbond.daily_twap"]["code"] == stale_twap_code)
            & (sources["market_cbond.daily_twap"]["trade_date"] == anchor)
        )
    ].copy()
    stale_twap_frame = _build({**sources, "market_cbond.daily_twap": stale_twap})
    twap_signals = [
        entry.signal
        for entry in dynamics.factor_mining_catalog()
        if entry.family in {"final_mark_execution_dislocation", "phase_conditioned_finalization"}
    ]
    barrier_signals = [
        entry.signal
        for entry in dynamics.factor_mining_catalog()
        if entry.family == "barrier_occupancy_hysteresis"
    ]
    assert stale_twap_frame.loc[(SCORE, CODES[0]), twap_signals].isna().all()
    assert stale_twap_frame.loc[(SCORE, CODES[0]), barrier_signals].notna().all()

    gapped_base_code = _bare(CODES[1])
    prior_day = sorted(sources["market_cbond.daily_price"]["trade_date"].unique())[-2]
    gapped_base = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == gapped_base_code)
            & (sources["market_cbond.daily_base"]["trade_date"] == prior_day)
        )
    ].copy()
    gapped_base_frame = _build({**sources, "market_cbond.daily_base": gapped_base})
    assert gapped_base_frame.loc[(SCORE, CODES[1]), barrier_signals].isna().all()
    assert gapped_base_frame.loc[(SCORE, CODES[1]), twap_signals].notna().all()


def test_missing_source_column_duplicate_rows_and_unknown_signal_fail_closed() -> None:
    kernel = dynamics.FactorMiningDailyMarkBarrierDynamicsV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={},
                params={"signal": "fmed_mark_execution_log_basis"},
            )
        )

    sources = _daily_sources()
    missing_twap = sources["market_cbond.daily_twap"].drop(columns=["twap_1442_1457"])
    with pytest.raises(KeyError, match="twap_1442_1457"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={**sources, "market_cbond.daily_twap": missing_twap},
                params={"signal": "fmed_mark_execution_log_basis"},
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
                daily_data={**sources, "market_cbond.daily_price": duplicate_price},
                params={"signal": "boh_edge_state_flip_rate20"},
            )
        )

    with pytest.raises(KeyError, match="unknown signal"):
        kernel.compute(FactorComputeContext(panel=_panel(), params={"signal": "unknown_daily_mark"}))


def test_phase_residual_evaluates_current_row_after_prior_only_training() -> None:
    rows = np.arange(dynamics._PHASE_TRAINING_DAYS, dtype="float64")
    predictors = np.column_stack((np.sin(rows / 5.0), np.cos(rows / 7.0), rows / 100.0))
    target = 0.002 + 0.8 * predictors[:, 0] - 0.5 * predictors[:, 1] + 0.3 * predictors[:, 2]
    training = np.column_stack((target, predictors))
    current = np.array([0.37, 0.2, -0.4, 0.51], dtype="float64")

    baseline = dynamics._one_phase_oos_residual(training, current)
    changed_target = current.copy()
    changed_target[0] += 0.25
    changed = dynamics._one_phase_oos_residual(training, changed_target)

    assert np.isfinite(baseline)
    assert changed - baseline == pytest.approx(0.25)
