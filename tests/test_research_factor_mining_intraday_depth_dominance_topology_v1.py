from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_intraday_depth_dominance_topology_v1 as topology
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
OTHER_BOND = "123001.SZ"


def _times() -> list[pd.Timedelta]:
    morning = [pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=10 * step) for step in range(12)]
    afternoon = [pd.Timedelta(hours=13) + pd.Timedelta(minutes=10 * step) for step in range(9)]
    return [*morning, *afternoon]


def _complete_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code_offset, code in enumerate((BOND, OTHER_BOND)):
        for sequence, clock in enumerate(_times()):
            bid_dominant = (2 * sequence + code_offset) % 5
            ask_dominant = (3 * sequence + 1 + code_offset) % 5
            row: dict[str, object] = {
                "dt": DT,
                "code": code,
                "seq": sequence,
                "trade_time": DT.normalize() + clock,
            }
            for level_index in range(5):
                level = level_index + 1
                row[f"bid_volume{level}"] = float(20 + level_index * 3 + ((sequence + level_index) % 5))
                row[f"ask_volume{level}"] = float(19 + level_index * 4 + ((2 * sequence + level_index) % 7))
            row[f"bid_volume{bid_dominant + 1}"] += 100.0
            row[f"ask_volume{ask_dominant + 1}"] += 110.0
            rows.append(row)
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=topology.KERNEL_NAME, params={"signal": entry.signal})
        for entry in topology.factor_mining_catalog()
    ]


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[:2].copy()
    prior["seq"] = prior["seq"] + 100
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior.loc[:, [column for column in prior.columns if column.startswith(("bid_volume", "ask_volume"))]] = 999_999.0

    lunch = frame.iloc[[2]].copy()
    lunch["seq"] = 200
    lunch["trade_time"] = DT.normalize() + pd.Timedelta(hours=12)
    lunch.loc[:, [column for column in lunch.columns if column.startswith(("bid_volume", "ask_volume"))]] = 888_888.0

    after_cutoff = frame.iloc[[3]].copy()
    after_cutoff["seq"] = 300
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_cutoff.loc[:, [column for column in after_cutoff.columns if column.startswith(("bid_volume", "ask_volume"))]] = 777_777.0

    contaminated = pd.concat([prior, frame, lunch, after_cutoff], ignore_index=True).set_index(["dt", "code", "seq"])
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    return contaminated


def test_depth_dominance_catalogue_has_exact_three_families_nine_signals_and_registry() -> None:
    entries = topology.factor_mining_catalog()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "depth_dominance_state_occupancy": 3,
        "depth_dominance_switch_topology": 3,
        "depth_dominance_transition_orientation": 3,
    }
    assert {entry.kernel for entry in entries} == {topology.KERNEL_NAME}
    assert FactorRegistry.get(topology.KERNEL_NAME) is topology.FactorMiningIntradayDepthDominanceTopologyV1
    assert set(topology.FORMULAS) == {entry.signal for entry in entries}
    assert not topology.FactorMiningIntradayDepthDominanceTopologyV1.requires_stock_panel
    assert not topology.FactorMiningIntradayDepthDominanceTopologyV1.requires_bond_stock_map
    assert topology.FactorMiningIntradayDepthDominanceTopologyV1.daily_requirements() == []


def test_depth_dominance_kernel_builds_finite_dt_code_contract() -> None:
    frame = build_factor_frame(_complete_panel(), _specs())

    assert frame.columns.tolist() == [entry.signal for entry in topology.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert frame.notna().all(axis=None)
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_depth_dominance_ignores_relabelled_prior_lunch_and_rows_after_1429() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    contaminated = build_factor_frame(_with_out_of_contract_rows(_complete_panel()), _specs())

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_depth_dominance_requires_all_five_levels_and_fails_closed_on_tied_maximum() -> None:
    panel = _complete_panel()
    with pytest.raises(KeyError, match="ask_volume5"):
        topology.FactorMiningIntradayDepthDominanceTopologyV1().compute(
            FactorComputeContext(panel=panel.drop(columns=["ask_volume5"]), params={"signal": "ddso_bid_dominant_level_entropy"})
        )

    tied = panel.copy()
    tied.loc[:, "bid_volume2"] = tied["bid_volume1"]
    tied.attrs["__build_day__"] = DT.date().isoformat()
    frame = build_factor_frame(tied, _specs())
    assert frame.isna().all(axis=None)
