from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_limit_pressure_temporal_topology_v1 as topology
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"


def _times() -> list[pd.Timedelta]:
    return [
        *[pd.Timedelta(hours=9, minutes=30 + 5 * index) for index in range(25)],
        *[pd.Timedelta(hours=13, minutes=5 * index) for index in range(18)],
        pd.Timedelta(hours=14, minutes=29),
    ]


def _complete_panel(*, code: str = BOND) -> pd.DataFrame:
    prices = np.array(
        [
            100.0,
            100.7,
            101.5,
            102.2,
            104.56,
            104.72,
            103.65,
            102.5,
            101.9,
            101.2,
            100.6,
            100.2,
            100.5,
            101.0,
            101.8,
            102.4,
            102.0,
            101.7,
            101.4,
            101.0,
            100.8,
            100.9,
            101.2,
            101.4,
            101.5,
            102.0,
            102.5,
            103.0,
            103.5,
            104.58,
            104.75,
            103.55,
            102.8,
            102.2,
            101.6,
            101.3,
            101.0,
            100.8,
            100.7,
            100.9,
            101.1,
            101.0,
            100.8,
            100.7,
        ],
        dtype="float64",
    )
    rows: list[dict[str, object]] = []
    for sequence, (clock, price) in enumerate(zip(_times(), prices, strict=True)):
        bid_depth = float(27 + ((sequence * 7 + 2) % 23))
        ask_depth = float(30 + ((sequence * 11 + 5) % 29))
        spread = 0.08 + 0.004 * (sequence % 4)
        rows.append(
            {
                "dt": DT,
                "code": code,
                "seq": sequence,
                "trade_time": DT.normalize() + clock,
                "last": float(price),
                "high_limited": 105.0,
                "low_limited": 95.0,
                "ask_price1": float(price + spread / 2.0),
                "bid_price1": float(price - spread / 2.0),
                "ask_volume1": ask_depth,
                "bid_volume1": bid_depth,
            }
        )
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=topology.KERNEL_NAME, params={"signal": entry.signal})
        for entry in topology.factor_mining_catalog()
    ]


def _build(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    return build_factor_frame(_complete_panel() if panel is None else panel, _specs())


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[[0]].copy()
    prior["seq"] = 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 10_000.0

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["last"] = 20_000.0

    late = frame.iloc[[-1]].copy()
    late["seq"] = 3_000
    late["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    late["last"] = 30_000.0

    out = pd.concat([prior, frame, preopen, late], ignore_index=True).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_limit_topology_catalogue_is_nonlock_and_panel_only() -> None:
    entries = topology.factor_mining_catalog()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "limit_nonlock_approach_entry_topology": 3,
        "limit_nonlock_approach_exit_sequence": 3,
        "limit_nonlock_approach_entry_liquidity": 3,
    }
    assert set(topology.FORMULAS) == {entry.signal for entry in entries}
    assert "exp_limit_near_touch_share" not in {entry.signal for entry in entries}
    assert FactorRegistry.get(topology.KERNEL_NAME) is topology.FactorMiningLimitPressureTemporalTopologyV1
    assert topology.FactorMiningLimitPressureTemporalTopologyV1.requires_stock_panel is False
    assert topology.FactorMiningLimitPressureTemporalTopologyV1.requires_bond_stock_map is False
    assert topology.FactorMiningLimitPressureTemporalTopologyV1.daily_requirements() == []


def test_limit_topology_builds_full_catalogue_without_inf() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in topology.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert int(frame.notna().sum(axis=1).iloc[0]) == 9
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_limit_topology_ignores_out_of_contract_rows() -> None:
    baseline = _build()
    contaminated = _build(_with_out_of_contract_rows(_complete_panel()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_limit_topology_returns_nan_when_only_actual_limit_locks_exist() -> None:
    locked = _complete_panel().reset_index()
    near = locked["last"] >= 104.5
    locked.loc[near, "last"] = 105.0
    locked.loc[near, "bid_price1"] = 105.0
    locked.loc[near, "ask_price1"] = 105.05
    panel = locked.set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    assert _build(panel).isna().all(axis=None)


def test_limit_topology_returns_nan_without_nonlock_approach_state() -> None:
    neutral = _complete_panel().reset_index()
    neutral["last"] = 100.0
    neutral["ask_price1"] = 100.05
    neutral["bid_price1"] = 99.95
    panel = neutral.set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    assert _build(panel).isna().all(axis=None)


@pytest.mark.parametrize("column, value", [("high_limited", 0.0), ("ask_price1", 0.0), ("bid_volume1", -1.0)])
def test_limit_topology_fails_closed_on_invalid_limit_or_l1_input(column: str, value: float) -> None:
    bad = _complete_panel().reset_index()
    bad.loc[10, column] = value
    panel = bad.set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    assert _build(panel).isna().all(axis=None)


def test_limit_topology_fails_closed_on_in_session_gap() -> None:
    gap = _complete_panel().reset_index()
    gap = gap.loc[
        ~gap["trade_time"].isin(
            [
                DT.normalize() + pd.Timedelta(hours=10, minutes=10),
                DT.normalize() + pd.Timedelta(hours=10, minutes=15),
                DT.normalize() + pd.Timedelta(hours=10, minutes=20),
                DT.normalize() + pd.Timedelta(hours=10, minutes=25),
                DT.normalize() + pd.Timedelta(hours=10, minutes=30),
            ]
        )
    ]
    panel = gap.set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    assert _build(panel).isna().all(axis=None)


def test_limit_topology_requires_fields_and_rejects_unknown_signal() -> None:
    missing = _complete_panel().drop(columns=["high_limited"])
    missing.attrs["__build_day__"] = DT.date().isoformat()
    with pytest.raises(KeyError, match="high_limited"):
        topology.FactorMiningLimitPressureTemporalTopologyV1().compute(
            FactorComputeContext(panel=missing, params={"signal": topology._ENTRY_TOPOLOGY_SIGNALS[0]})
        )
    with pytest.raises(KeyError, match="unknown signal"):
        topology.FactorMiningLimitPressureTemporalTopologyV1().compute(
            FactorComputeContext(panel=_complete_panel(), params={"signal": "lpt_unknown"})
        )
