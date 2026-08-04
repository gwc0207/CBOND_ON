from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_intraday_trade_grid_topology_v1 as grid
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
OTHER_BOND = "123001.SZ"


def _times() -> list[pd.Timedelta]:
    morning = [pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=7 * step) for step in range(16)]
    afternoon = [pd.Timedelta(hours=13) + pd.Timedelta(minutes=7 * step) for step in range(12)]
    return [*morning, *afternoon]


def _price_steps() -> list[float]:
    morning = [0.10, 0.0, 0.0, -0.30, 0.10, -0.20, 0.10, -0.10, 0.0, 0.0, 0.30, -0.10, 0.20, -0.10, 0.10]
    lunch_gap = [0.0]
    afternoon = [0.20, 0.0, 0.0, -0.40, -0.10, -0.20, 0.30, 0.10, 0.20, -0.30, -0.10]
    return [*morning, *lunch_gap, *afternoon]


def _complete_panel() -> pd.DataFrame:
    times = _times()
    base_steps = np.asarray(_price_steps(), dtype="float64")
    assert len(base_steps) == len(times) - 1
    rows: list[dict[str, object]] = []
    for code_offset, code in enumerate((BOND, OTHER_BOND)):
        scale = 1.0 + 0.25 * code_offset
        signed_steps = base_steps * scale * (-1.0 if code_offset else 1.0)
        prices = 100.0 + 5.0 * code_offset + np.concatenate(([0.0], np.cumsum(signed_steps)))
        trades = np.arange(len(times), dtype="float64")
        for sequence, (clock, last, count) in enumerate(zip(times, prices, trades, strict=True)):
            rows.append(
                {
                    "dt": DT,
                    "code": code,
                    "seq": sequence,
                    "trade_time": DT.normalize() + clock,
                    "last": float(last),
                    "num_trades": float(count),
                }
            )
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=grid.KERNEL_NAME, params={"signal": entry.signal})
        for entry in grid.factor_mining_catalog()
    ]


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[:2].copy()
    prior["seq"] = prior["seq"] + 100
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 1.0
    prior["num_trades"] = 1_000_000.0

    lunch = frame.iloc[[2]].copy()
    lunch["seq"] = 200
    lunch["trade_time"] = DT.normalize() + pd.Timedelta(hours=12)
    lunch["last"] = 1.0
    lunch["num_trades"] = 2_000_000.0

    after_cutoff = frame.iloc[[3]].copy()
    after_cutoff["seq"] = 300
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_cutoff["last"] = 1.0
    after_cutoff["num_trades"] = 3_000_000.0

    contaminated = pd.concat([prior, frame, lunch, after_cutoff], ignore_index=True).set_index(["dt", "code", "seq"])
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    return contaminated


def test_trade_grid_catalogue_has_exact_three_families_nine_signals_and_registry() -> None:
    entries = grid.factor_mining_catalog()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "trade_grid_phase_scale_topology": 3,
        "trade_grid_zero_spell_exit": 3,
        "trade_grid_transition_amplitude": 3,
    }
    assert {entry.kernel for entry in entries} == {grid.KERNEL_NAME}
    assert FactorRegistry.get(grid.KERNEL_NAME) is grid.FactorMiningIntradayTradeGridTopologyV1
    assert set(grid.FORMULAS) == {entry.signal for entry in entries}
    assert not grid.FactorMiningIntradayTradeGridTopologyV1.requires_stock_panel
    assert not grid.FactorMiningIntradayTradeGridTopologyV1.requires_bond_stock_map
    assert grid.FactorMiningIntradayTradeGridTopologyV1.daily_requirements() == []


def test_trade_grid_kernel_builds_finite_dt_code_contract() -> None:
    frame = build_factor_frame(_complete_panel(), _specs())

    assert frame.columns.tolist() == [entry.signal for entry in grid.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND), (DT, OTHER_BOND)]
    assert frame.notna().all(axis=None)
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_trade_grid_ignores_relabelled_prior_lunch_and_rows_after_1429() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    contaminated = build_factor_frame(_with_out_of_contract_rows(_complete_panel()), _specs())

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_trade_grid_fails_closed_on_counter_reset_and_requires_explicit_fields() -> None:
    panel = _complete_panel()
    with pytest.raises(KeyError, match="num_trades"):
        grid.FactorMiningIntradayTradeGridTopologyV1().compute(
            FactorComputeContext(panel=panel.drop(columns=["num_trades"]), params={"signal": "tgps_large_step_late_tilt"})
        )

    reset = panel.copy()
    reset.iloc[10, reset.columns.get_loc("num_trades")] = -1.0
    reset.attrs["__build_day__"] = DT.date().isoformat()
    frame = build_factor_frame(reset, _specs())
    assert frame.loc[(DT, BOND)].isna().all()
