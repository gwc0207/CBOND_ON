from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_quote_execution_dynamics_v1 as dynamics
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
        pd.Timedelta(hours=10, minutes=12),
        pd.Timedelta(hours=10, minutes=25),
        pd.Timedelta(hours=10, minutes=38),
        pd.Timedelta(hours=10, minutes=52),
        pd.Timedelta(hours=11, minutes=8),
        pd.Timedelta(hours=11, minutes=22),
        pd.Timedelta(hours=13),
        pd.Timedelta(hours=13, minutes=14),
        pd.Timedelta(hours=13, minutes=29),
        pd.Timedelta(hours=13, minutes=45),
        pd.Timedelta(hours=14, minutes=2),
        pd.Timedelta(hours=14, minutes=17),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _locations() -> list[float]:
    return [0.0, 0.25, 1.25, 1.45, -1.20, -1.55, 0.55, -1.10, 1.20, 0.0, 1.80, -1.30, -1.15, 0.35, 1.35, 1.55, -1.70, 1.10]


def _complete_panel(*, code: str = BOND) -> pd.DataFrame:
    locations = _locations()
    cumulative = 100.0
    rows: list[dict[str, object]] = []
    for seq, (clock, location) in enumerate(zip(_times(), locations, strict=True)):
        mid = 100.0 + 0.03 * seq
        spread = 0.16 + 0.01 * (seq % 3)
        ask = mid + spread / 2.0
        bid = mid - spread / 2.0
        if seq == 0:
            last = mid
        else:
            prior_mid = 100.0 + 0.03 * (seq - 1)
            prior_spread = 0.16 + 0.01 * ((seq - 1) % 3)
            last = prior_mid + location * prior_spread / 2.0
        cumulative += float(1 + ((3 * seq + 2) % 5))
        rows.append(
            {
                "dt": DT,
                "code": code,
                "seq": seq,
                "trade_time": DT.normalize() + clock,
                "last": last,
                "ask_price1": ask,
                "bid_price1": bid,
                "num_trades": cumulative,
            }
        )
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=dynamics.KERNEL_NAME, params={"signal": entry.signal})
        for entry in dynamics.factor_mining_catalog()
    ]


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[[0]].copy()
    prior["seq"] = 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["num_trades"] = 1_000_000.0
    prior["last"] = 1.0

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["num_trades"] = 2_000_000.0
    preopen["last"] = 1.0

    after_1429 = frame.iloc[[-1]].copy()
    after_1429["seq"] = 3_000
    after_1429["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_1429["num_trades"] = 3_000_000.0
    after_1429["last"] = 1.0

    at_1430 = frame.iloc[[-1]].copy()
    at_1430["seq"] = 3_001
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    at_1430["num_trades"] = 4_000_000.0
    at_1430["last"] = 1.0

    out = pd.concat([prior, frame, preopen, after_1429, at_1430], ignore_index=True).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_quote_execution_catalogue_registration_and_context_contract() -> None:
    entries = dynamics.factor_mining_catalog()

    assert len(entries) == 6
    assert len({entry.signal for entry in entries}) == 6
    assert Counter(entry.family for entry in entries) == {
        "quote_execution_location_shape": 3,
        "quote_side_sequence_dynamics": 3,
    }
    assert FactorRegistry.get(dynamics.KERNEL_NAME) is dynamics.FactorMiningQuoteExecutionDynamicsV1
    assert set(dynamics.FORMULAS) == {entry.signal for entry in entries}
    assert not dynamics.FactorMiningQuoteExecutionDynamicsV1.requires_stock_panel
    assert not dynamics.FactorMiningQuoteExecutionDynamicsV1.requires_bond_stock_map
    assert dynamics.FactorMiningQuoteExecutionDynamicsV1.daily_requirements() == []


def test_quote_execution_kernel_builds_six_finite_prior_quote_signals_without_inf() -> None:
    frame = build_factor_frame(_complete_panel(), _specs())

    assert frame.columns.tolist() == [entry.signal for entry in dynamics.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert int(frame.notna().sum(axis=1).iloc[0]) == 6
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame.loc[(DT, BOND), "qed_prior_quote_location_dispersion"] > 0.0
    assert frame.loc[(DT, BOND), "qed_prior_quote_inside_spread_share"] > 0.0
    assert frame.loc[(DT, BOND), "qed_prior_quote_tail_penetration"] > 0.0


def test_quote_execution_uses_strictly_prior_quote_not_terminal_current_quote() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    changed = _complete_panel().reset_index()
    changed.loc[changed.index[-1], "ask_price1"] += 10.0
    changed.loc[changed.index[-1], "bid_price1"] += 10.0
    changed_panel = changed.set_index(["dt", "code", "seq"])
    changed_panel.attrs["__build_day__"] = DT.date().isoformat()

    result = build_factor_frame(changed_panel, _specs())

    pd.testing.assert_frame_equal(result, baseline, check_exact=True)


def test_quote_execution_ignores_nonphysical_preopen_and_after_cutoff_rows() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    contaminated = build_factor_frame(_with_out_of_contract_rows(_complete_panel()), _specs())

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_quote_execution_counter_reset_fails_closed_without_inf() -> None:
    panel = _complete_panel().reset_index()
    panel.loc[panel.index[-1], "num_trades"] = 1.0
    reset_panel = panel.set_index(["dt", "code", "seq"])
    reset_panel.attrs["__build_day__"] = DT.date().isoformat()

    frame = build_factor_frame(reset_panel, _specs())

    assert frame.isna().all(axis=None)
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_quote_execution_missing_field_and_no_events_fail_closed() -> None:
    missing = _complete_panel().drop(columns=["last"])
    missing.attrs["__build_day__"] = DT.date().isoformat()
    missing_result = build_factor_frame(missing, _specs())

    no_events = _complete_panel().reset_index()
    no_events["num_trades"] = 100.0
    no_events_panel = no_events.set_index(["dt", "code", "seq"])
    no_events_panel.attrs["__build_day__"] = DT.date().isoformat()
    no_events_result = build_factor_frame(no_events_panel, _specs())

    assert missing_result.isna().all(axis=None)
    assert no_events_result.isna().all(axis=None)


def test_quote_execution_rejects_unknown_signal() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        dynamics.FactorMiningQuoteExecutionDynamicsV1().compute(
            FactorComputeContext(
                panel=_complete_panel(),
                params={"signal": "qed_unknown"},
            )
        )
