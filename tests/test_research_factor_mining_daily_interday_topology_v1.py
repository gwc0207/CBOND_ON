from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_daily_interday_topology_v1 as topology
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


def _basis(day_index: int, code_index: int) -> float:
    stock_return = 0.004 * np.sin((day_index + 1.2 * code_index) / 4.7)
    stock_volatility = 0.16 + 0.018 * np.cos((day_index + 0.5 * code_index) / 6.3)
    return float(
        0.0006
        + 0.46 * stock_return
        - 0.035 * (stock_volatility - 0.16)
        + 0.0012 * np.sin((day_index + 2.1 * code_index) / 2.9)
    )


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=66)
    price_rows: list[dict[str, object]] = []
    twap_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        for day_index, day in enumerate(days):
            phase = float(day_index + 1.7 * code_index)
            stock_prev = 10.0 + 0.08 * code_index + 0.006 * day_index
            stock_return = 0.004 * np.sin((day_index + 1.2 * code_index) / 4.7)
            stock_close = stock_prev * np.exp(stock_return)
            stock_volatility = 0.16 + 0.018 * np.cos((day_index + 0.5 * code_index) / 6.3)

            execution = 100.0 + 0.7 * code_index + 0.09 * day_index
            basis = _basis(day_index, code_index)
            close = execution * np.exp(basis)
            cb_prior = 99.0 + 0.7 * code_index + 0.08 * day_index
            cb_adjusted_prior = cb_prior * (1.0 + 0.0002 * np.sin(phase / 5.1))
            prior_basis = _basis(day_index - 1, code_index)
            opening_return = (
                0.0002
                + 0.62 * prior_basis
                + 0.0007 * np.cos((day_index + 1.4 * code_index) / 3.7)
            )
            opening = cb_adjusted_prior * np.exp(opening_return)

            pure_redemption = 79.0 + 0.1 * code_index + 0.018 * day_index
            floor_log = 0.13 + 0.026 * np.sin((day_index + code_index) / 5.7)
            cb_close = pure_redemption * np.exp(floor_log)
            put = 7.0 + 0.035 * code_index + 0.025 * np.sin(phase / 6.1)
            call = 13.0 + 0.035 * code_index + 0.025 * np.cos(phase / 7.3)
            year_to_mat = 2.8 - 0.0035 * day_index + 0.035 * code_index

            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "close_price": close,
                    "prev_close_price": cb_prior,
                    "act_prev_close_price": cb_adjusted_prior,
                }
            )
            twap_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "twap_0930_0935": opening,
                    "twap_1442_1457": execution,
                }
            )
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "stk_prev_close_price": stock_prev,
                    "stk_act_prev_close_price": stock_prev * (1.0 + 0.0001 * np.cos(phase / 4.0)),
                    "stk_close_price": stock_close,
                    "stock_volatility": stock_volatility,
                    "cb_close_price": cb_close,
                    "pure_redemption_value": pure_redemption,
                    "cb_put_price": put,
                    "cb_call_price": call,
                    "stock_close_price": stock_close,
                    "year_to_mat": year_to_mat,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_twap": pd.DataFrame(twap_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=topology.KERNEL_NAME, params={"signal": entry.signal})
        for entry in topology.factor_mining_catalog()
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


def test_catalogue_has_three_families_and_exact_context_contracts() -> None:
    entries = topology.factor_mining_catalog()
    all_requirements = topology.FactorMiningDailyInterdayTopologyV1.daily_requirements()
    interday_requirements = topology.FactorMiningDailyInterdayTopologyV1.daily_requirements(
        {"signal": "ift_centered_basis_forecast40"}
    )
    stock_requirements = topology.FactorMiningDailyInterdayTopologyV1.daily_requirements(
        {"signal": "scmr_mark_stock_oos_residual40"}
    )
    topology_requirements = topology.FactorMiningDailyInterdayTopologyV1.daily_requirements(
        {"signal": "cft_call_floor_oos_residual40"}
    )

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "interday_finalization_transmission": 3,
        "stock_conditioned_mark_residual": 3,
        "contract_floor_payoff_topology": 3,
    }
    assert set(topology.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(topology.KERNEL_NAME) is topology.FactorMiningDailyInterdayTopologyV1
    assert topology.FactorMiningDailyInterdayTopologyV1.__name__ not in factor_operators.__all__

    assert [item.source for item in interday_requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_twap",
    ]
    assert interday_requirements[0].columns == (
        "exchange_code",
        "close_price",
        "prev_close_price",
        "act_prev_close_price",
    )
    assert interday_requirements[1].columns == (
        "exchange_code",
        "twap_0930_0935",
        "twap_1442_1457",
    )
    assert [item.source for item in stock_requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_twap",
        "market_cbond.daily_base",
    ]
    assert stock_requirements[2].columns == (
        "exchange_code",
        "stk_prev_close_price",
        "stk_act_prev_close_price",
        "stk_close_price",
        "stock_volatility",
    )
    assert [item.source for item in topology_requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_base",
    ]
    assert topology_requirements[1].columns == (
        "exchange_code",
        "cb_close_price",
        "pure_redemption_value",
        "cb_put_price",
        "cb_call_price",
        "stock_close_price",
        "year_to_mat",
        "stock_volatility",
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

    assert frame.columns.tolist() == [entry.signal for entry in topology.factor_mining_catalog()]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert (frame.notna().sum(axis=0) == len(CODES)).all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame["ift_centered_basis_forecast40"].abs().sum() > 0.0
    assert frame["scmr_mark_stock_oos_residual40"].abs().sum() > 0.0
    assert frame["cft_call_floor_oos_residual40"].abs().sum() > 0.0


def test_score_day_and_future_daily_rows_are_ignored_exactly() -> None:
    baseline = _build()
    contaminated = _build(_contaminate(_daily_sources()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_anchor_basis_is_not_a_training_pair_for_interday_beta_or_hit_rate() -> None:
    sources = _daily_sources()
    baseline = _build(sources)
    anchor = sources["market_cbond.daily_price"]["trade_date"].max()
    moved_price = sources["market_cbond.daily_price"].copy()
    moved_price.loc[
        (moved_price["code"] == _bare(CODES[0])) & (moved_price["trade_date"] == anchor),
        "close_price",
    ] *= 1.25
    changed = _build({**sources, "market_cbond.daily_price": moved_price})

    row = (SCORE, CODES[0])
    assert changed.loc[row, "ift_transmission_beta40"] == pytest.approx(
        baseline.loc[row, "ift_transmission_beta40"]
    )
    assert changed.loc[row, "ift_direction_hit_rate40"] == pytest.approx(
        baseline.loc[row, "ift_direction_hit_rate40"]
    )
    assert changed.loc[row, "ift_centered_basis_forecast40"] != pytest.approx(
        baseline.loc[row, "ift_centered_basis_forecast40"]
    )


def test_stale_anchor_and_interior_gaps_fail_closed_only_for_affected_families() -> None:
    sources = _daily_sources()
    anchor = sources["market_cbond.daily_price"]["trade_date"].max()

    stale_twap_code = _bare(CODES[0])
    stale_twap = sources["market_cbond.daily_twap"].loc[
        ~(
            (sources["market_cbond.daily_twap"]["code"] == stale_twap_code)
            & (sources["market_cbond.daily_twap"]["trade_date"] == anchor)
        )
    ].copy()
    stale_frame = _build({**sources, "market_cbond.daily_twap": stale_twap})
    twap_families = {"interday_finalization_transmission", "stock_conditioned_mark_residual"}
    twap_signals = [entry.signal for entry in topology.factor_mining_catalog() if entry.family in twap_families]
    contract_signals = [
        entry.signal
        for entry in topology.factor_mining_catalog()
        if entry.family == "contract_floor_payoff_topology"
    ]
    assert stale_frame.loc[(SCORE, CODES[0]), twap_signals].isna().all()
    assert stale_frame.loc[(SCORE, CODES[0]), contract_signals].notna().all()

    gapped_base_code = _bare(CODES[1])
    prior_day = sorted(sources["market_cbond.daily_price"]["trade_date"].unique())[-2]
    gapped_base = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == gapped_base_code)
            & (sources["market_cbond.daily_base"]["trade_date"] == prior_day)
        )
    ].copy()
    gapped_frame = _build({**sources, "market_cbond.daily_base": gapped_base})
    base_families = {"stock_conditioned_mark_residual", "contract_floor_payoff_topology"}
    base_signals = [entry.signal for entry in topology.factor_mining_catalog() if entry.family in base_families]
    interday_signals = [
        entry.signal
        for entry in topology.factor_mining_catalog()
        if entry.family == "interday_finalization_transmission"
    ]
    assert gapped_frame.loc[(SCORE, CODES[1]), base_signals].isna().all()
    assert gapped_frame.loc[(SCORE, CODES[1]), interday_signals].notna().all()


def test_missing_source_column_duplicate_rows_and_unknown_signal_fail_closed() -> None:
    kernel = topology.FactorMiningDailyInterdayTopologyV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={},
                params={"signal": "ift_centered_basis_forecast40"},
            )
        )

    sources = _daily_sources()
    missing_base = sources["market_cbond.daily_base"].drop(columns=["stock_volatility"])
    with pytest.raises(KeyError, match="stock_volatility"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={**sources, "market_cbond.daily_base": missing_base},
                params={"signal": "scmr_mark_stock_oos_residual40"},
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
                params={"signal": "ift_centered_basis_forecast40"},
            )
        )

    with pytest.raises(KeyError, match="unknown signal"):
        kernel.compute(FactorComputeContext(panel=_panel(), params={"signal": "unknown_interday"}))


def test_current_oos_target_shift_changes_residual_one_for_one_without_refitting() -> None:
    rows = np.arange(topology._TRAINING_DAYS, dtype="float64")
    predictors = np.column_stack((np.sin(rows / 4.0), np.cos(rows / 6.0)))
    target = 0.003 + 0.7 * predictors[:, 0] - 0.4 * predictors[:, 1]
    training = np.column_stack((target, predictors))
    current = np.array([0.17, 0.2, -0.3], dtype="float64")

    baseline = topology._one_oos_residual(training, current)
    changed_target = current.copy()
    changed_target[0] += 0.25
    changed = topology._one_oos_residual(training, changed_target)

    assert np.isfinite(baseline)
    assert changed - baseline == pytest.approx(0.25)
