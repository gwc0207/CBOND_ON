from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_intraday_execution_discreteness_v1 as discreteness
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=9, minutes=35),
        pd.Timedelta(hours=9, minutes=42),
        pd.Timedelta(hours=9, minutes=50),
        pd.Timedelta(hours=10),
        pd.Timedelta(hours=10, minutes=10),
        pd.Timedelta(hours=10, minutes=22),
        pd.Timedelta(hours=10, minutes=35),
        pd.Timedelta(hours=10, minutes=51),
        pd.Timedelta(hours=11, minutes=8),
        pd.Timedelta(hours=11, minutes=23),
        pd.Timedelta(hours=13),
        pd.Timedelta(hours=13, minutes=12),
        pd.Timedelta(hours=13, minutes=27),
        pd.Timedelta(hours=13, minutes=45),
        pd.Timedelta(hours=14, minutes=2),
        pd.Timedelta(hours=14, minutes=17),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _prices() -> list[float]:
    return [
        100.00,
        100.00,
        100.00,
        100.01,
        100.03,
        100.03,
        100.03,
        100.02,
        100.02,
        100.05,
        100.04,
        100.04,
        100.03,
        100.03,
        100.05,
        100.05,
        100.05,
        100.02,
    ]


def _complete_panel(*, code: str = BOND) -> pd.DataFrame:
    cumulative = 100.0
    rows: list[dict[str, object]] = []
    prices = _prices()
    for seq, (clock, last) in enumerate(zip(_times(), prices, strict=True)):
        if seq > 0 and seq != len(prices) - 1:
            cumulative += float(1 + ((3 * seq + 2) % 5))
        rows.append(
            {
                "dt": DT,
                "code": code,
                "seq": seq,
                "trade_time": DT.normalize() + clock,
                "last": last,
                "num_trades": cumulative,
            }
        )
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=discreteness.KERNEL_NAME, params={"signal": entry.signal})
        for entry in discreteness.factor_mining_catalog()
    ]


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[[0]].copy()
    prior["seq"] = 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 1.0
    prior["num_trades"] = 1_000_000.0

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["last"] = 1.0
    preopen["num_trades"] = 2_000_000.0

    after_1429 = frame.iloc[[-1]].copy()
    after_1429["seq"] = 3_000
    after_1429["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_1429["last"] = 1.0
    after_1429["num_trades"] = 3_000_000.0

    at_1430 = frame.iloc[[-1]].copy()
    at_1430["seq"] = 3_001
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    at_1430["last"] = 1.0
    at_1430["num_trades"] = 4_000_000.0

    out = pd.concat([prior, frame, preopen, after_1429, at_1430], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_execution_discreteness_catalogue_registration_and_context_contract() -> None:
    entries = discreteness.factor_mining_catalog()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "execution_step_multiplicity_geometry": 3,
        "execution_nonzero_direction_topology": 3,
        "execution_stationary_moving_regime": 3,
    }
    assert FactorRegistry.get(discreteness.KERNEL_NAME) is discreteness.FactorMiningIntradayExecutionDiscretenessV1
    assert set(discreteness.FORMULAS) == {entry.signal for entry in entries}
    assert not discreteness.FactorMiningIntradayExecutionDiscretenessV1.requires_stock_panel
    assert not discreteness.FactorMiningIntradayExecutionDiscretenessV1.requires_bond_stock_map
    assert discreteness.FactorMiningIntradayExecutionDiscretenessV1.daily_requirements() == []


def test_execution_discreteness_builds_nine_finite_signals_without_inf() -> None:
    frame = build_factor_frame(_complete_panel(), _specs())

    assert frame.columns.tolist() == [entry.signal for entry in discreteness.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert int(frame.notna().sum(axis=1).iloc[0]) == 9
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert 0.0 < frame.loc[(DT, BOND), "execdisc_step_unit_share"] < 1.0
    assert frame.loc[(DT, BOND), "execdisc_step_multiplicity_entropy"] > 0.0
    assert frame.loc[(DT, BOND), "execdisc_step_large_multiple_share"] > 0.0
    assert frame.loc[(DT, BOND), "execdisc_direction_reversal_rate"] > 0.0
    assert frame.loc[(DT, BOND), "execdisc_stationary_moving_mean_length_ratio"] > 0.0


def test_execution_discreteness_uses_only_trade_confirmed_intervals() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    changed = _complete_panel().reset_index()
    changed.loc[changed.index[-1], "last"] = 150.0
    changed_panel = changed.set_index(["dt", "code", "seq"])
    changed_panel.attrs["__build_day__"] = DT.date().isoformat()

    result = build_factor_frame(changed_panel, _specs())

    pd.testing.assert_frame_equal(result, baseline, check_exact=True)


def test_execution_discreteness_ignores_nonphysical_preopen_and_after_cutoff_rows() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    contaminated = build_factor_frame(_with_out_of_contract_rows(_complete_panel()), _specs())

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_execution_discreteness_counter_reset_or_duplicate_timestamp_fails_closed() -> None:
    reset = _complete_panel().reset_index()
    reset.loc[reset.index[10], "num_trades"] = 1.0
    reset_panel = reset.set_index(["dt", "code", "seq"])
    reset_panel.attrs["__build_day__"] = DT.date().isoformat()
    reset_result = build_factor_frame(reset_panel, _specs())

    duplicate = _complete_panel().reset_index()
    duplicate.loc[duplicate.index[8], "trade_time"] = duplicate.loc[duplicate.index[7], "trade_time"]
    duplicate_panel = duplicate.set_index(["dt", "code", "seq"])
    duplicate_panel.attrs["__build_day__"] = DT.date().isoformat()
    duplicate_result = build_factor_frame(duplicate_panel, _specs())

    assert reset_result.isna().all(axis=None)
    assert duplicate_result.isna().all(axis=None)
    assert not np.isinf(reset_result.to_numpy(dtype="float64")).any()
    assert not np.isinf(duplicate_result.to_numpy(dtype="float64")).any()


def test_execution_discreteness_missing_field_or_no_nonzero_step_fails_closed() -> None:
    missing = _complete_panel().drop(columns=["last"])
    missing.attrs["__build_day__"] = DT.date().isoformat()
    missing_result = build_factor_frame(missing, _specs())

    flat = _complete_panel().reset_index()
    flat["last"] = 100.0
    flat_panel = flat.set_index(["dt", "code", "seq"])
    flat_panel.attrs["__build_day__"] = DT.date().isoformat()
    flat_result = build_factor_frame(flat_panel, _specs())

    assert missing_result.isna().all(axis=None)
    assert flat_result.isna().all(axis=None)


def test_stationary_run_metrics_are_not_price_change_rate_or_longest_run() -> None:
    left = np.asarray([False, False, True, False, False, True, False, True], dtype=bool)
    right = np.asarray([False, False, True, False, True, False, True, False], dtype=bool)
    sessions = np.zeros(len(left), dtype="int8")

    assert float(np.mean(left)) == float(np.mean(right))
    left_states, left_lengths = discreteness._state_runs(left)
    right_states, right_lengths = discreteness._state_runs(right)
    assert int(left_lengths[~left_states].max()) == int(right_lengths[~right_states].max()) == 2
    left_metrics = discreteness._stationary_metrics(left, sessions)
    right_metrics = discreteness._stationary_metrics(right, sessions)

    assert left_metrics["execdisc_stationary_run_concentration"] != pytest.approx(
        right_metrics["execdisc_stationary_run_concentration"]
    )
    assert left_metrics["execdisc_stationary_moving_mean_length_ratio"] != pytest.approx(
        right_metrics["execdisc_stationary_moving_mean_length_ratio"]
    )


def test_execution_discreteness_rejects_unknown_signal() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        discreteness.FactorMiningIntradayExecutionDiscretenessV1().compute(
            FactorComputeContext(
                panel=_complete_panel(),
                params={"signal": "execdisc_unknown"},
            )
        )
