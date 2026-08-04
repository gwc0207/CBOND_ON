from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_intraday_queue_state_v1 as queue
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
        pd.Timedelta(hours=11, minutes=30),
        pd.Timedelta(hours=13),
        pd.Timedelta(hours=13, minutes=9),
        pd.Timedelta(hours=13, minutes=18),
        pd.Timedelta(hours=13, minutes=28),
        pd.Timedelta(hours=13, minutes=39),
        pd.Timedelta(hours=13, minutes=50),
        pd.Timedelta(hours=14),
        pd.Timedelta(hours=14, minutes=8),
        pd.Timedelta(hours=14, minutes=16),
        pd.Timedelta(hours=14, minutes=23),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _complete_panel(*, code: str = BOND) -> pd.DataFrame:
    base_mid = 100.0 + np.array(
        [0.00, 0.08, -0.04, 0.13, 0.05, 0.16, 0.02, 0.20, 0.11, 0.25, 0.17, 0.29,
         0.22, 0.35, 0.28, 0.41, 0.33, 0.48, 0.40, 0.56, 0.49, 0.65, 0.72],
        dtype="float64",
    )
    trade_steps = np.array(
        [0, 3, 2, 4, 1, 3, 5, 2, 4, 1, 6, 2, 3, 5, 1, 4, 2, 6, 3, 5, 1, 4, 3],
        dtype="float64",
    )
    trade_counts = np.cumsum(trade_steps)
    rows: list[dict[str, object]] = []
    previous_ask = float(base_mid[0] + 0.04)
    previous_bid = float(base_mid[0] - 0.04)
    for sequence, clock in enumerate(_times()):
        mid = float(base_mid[sequence])
        spread = float(0.07 + 0.01 * (sequence % 3))
        ask1 = mid + spread / 2.0
        bid1 = mid - spread / 2.0
        if sequence == 0:
            last = mid
        elif sequence % 3 == 0:
            last = previous_ask
        elif sequence % 3 == 1:
            last = previous_bid
        else:
            last = mid
        if sequence == len(_times()) - 1:
            last = 105.0
            bid1 = 105.0
            ask1 = 105.04
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": sequence,
            "trade_time": DT.normalize() + clock,
            "last": float(last),
            "num_trades": float(trade_counts[sequence]),
            "high_limited": 105.0,
            "low_limited": 95.0,
            "ask_price1": ask1,
            "bid_price1": bid1,
            "ask_volume1": float(42 + ((3 * sequence + 5) % 17)),
            "bid_volume1": float(47 + ((5 * sequence + 2) % 19)),
        }
        for level in range(2, 6):
            row[f"ask_price{level}"] = ask1 + 0.01 * level
            row[f"bid_price{level}"] = bid1 - 0.01 * level
            row[f"ask_volume{level}"] = float(35 + level * 6 + ((sequence * (level + 1)) % 13))
            row[f"bid_volume{level}"] = float(37 + level * 7 + ((2 * sequence * level) % 11))
        rows.append(row)
        previous_ask = ask1
        previous_bid = bid1
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=queue.KERNEL_NAME, params={"signal": entry.signal})
        for entry in queue.factor_mining_intraday_queue_state_v1_catalog()
    ]


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[:3].copy()
    prior["seq"] = prior["seq"] + 100
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 1.0
    prior["num_trades"] = 1_000_000.0

    after_cutoff = frame.iloc[[-1]].copy()
    after_cutoff["seq"] = 1000
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_cutoff["last"] = 1.0
    after_cutoff["num_trades"] = 2_000_000.0

    cutoff = frame.iloc[[-1]].copy()
    cutoff["seq"] = 1001
    cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    cutoff["last"] = 1.0
    cutoff["num_trades"] = 3_000_000.0

    lunch = frame.iloc[[0]].copy()
    lunch["seq"] = 1002
    lunch["trade_time"] = DT.normalize() + pd.Timedelta(hours=12)
    lunch["last"] = 1.0

    contaminated = pd.concat([prior, frame, after_cutoff, cutoff, lunch], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    return contaminated


def test_queue_state_catalogue_has_four_three_signal_families_and_registry() -> None:
    entries = queue.factor_mining_intraday_queue_state_v1_catalog()

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "limit_queue_lock_state_machine": 3,
        "passive_queue_lifecycle_after_trade": 3,
        "trade_at_quote_initiation_state": 3,
        "depth_layer_update_cascade": 3,
    }
    assert {entry.kernel for entry in entries} == {queue.KERNEL_NAME}
    assert FactorRegistry.get(queue.KERNEL_NAME) is queue.FactorMiningIntradayQueueStateV1
    assert set(queue.FORMULAS) == {entry.signal for entry in entries}
    assert not queue.FactorMiningIntradayQueueStateV1.requires_stock_panel
    assert not queue.FactorMiningIntradayQueueStateV1.requires_bond_stock_map
    assert queue.FactorMiningIntradayQueueStateV1.daily_requirements() == []


def test_queue_state_kernel_builds_dt_code_contract_without_inf() -> None:
    frame = build_factor_frame(_complete_panel(), _specs())

    expected_columns = [entry.signal for entry in queue.factor_mining_intraday_queue_state_v1_catalog()]
    assert frame.columns.tolist() == expected_columns
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert int(frame.notna().sum(axis=1).iloc[0]) == 12


def test_queue_state_ignores_relabelled_prior_lunch_and_rows_after_1429() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    contaminated = build_factor_frame(_with_out_of_contract_rows(_complete_panel()), _specs())

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_queue_state_counter_reset_fails_closed_for_trade_conditioned_families() -> None:
    panel = _complete_panel().copy()
    panel.iloc[10, panel.columns.get_loc("num_trades")] = panel.iloc[9, panel.columns.get_loc("num_trades")] - 1.0
    panel.attrs["__build_day__"] = DT.date().isoformat()
    entries = queue.factor_mining_intraday_queue_state_v1_catalog()
    trade_specs = [
        spec
        for spec, entry in zip(_specs(), entries, strict=True)
        if entry.family in {"passive_queue_lifecycle_after_trade", "trade_at_quote_initiation_state"}
    ]

    frame = build_factor_frame(panel, trade_specs)

    assert frame.index.tolist() == [(DT, BOND)]
    assert frame.isna().all(axis=None)


def test_queue_state_invalid_l1_book_fails_closed_only_for_affected_family() -> None:
    panel = _complete_panel().copy()
    panel.iloc[7, panel.columns.get_loc("ask_price1")] = 1.0
    panel.iloc[7, panel.columns.get_loc("bid_price1")] = 999.0
    panel.attrs["__build_day__"] = DT.date().isoformat()
    entries = queue.factor_mining_intraday_queue_state_v1_catalog()
    top_book_specs = [
        spec
        for spec, entry in zip(_specs(), entries, strict=True)
        if entry.family == "trade_at_quote_initiation_state"
    ]

    frame = build_factor_frame(panel, top_book_specs)

    assert frame.index.tolist() == [(DT, BOND)]
    assert frame.isna().all(axis=None)


def test_queue_state_requires_explicit_family_columns_without_fallback() -> None:
    panel = _complete_panel().drop(columns=["num_trades"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    with pytest.raises(KeyError, match="num_trades"):
        queue.FactorMiningIntradayQueueStateV1().compute(
            FactorComputeContext(panel=panel, params={"signal": "taqi_buy_quote_initiation_share"})
        )


def test_queue_state_unknown_signal_is_explicit_error() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        queue.FactorMiningIntradayQueueStateV1().compute(
            FactorComputeContext(panel=_complete_panel(), params={"signal": "not_a_queue_signal"})
        )
