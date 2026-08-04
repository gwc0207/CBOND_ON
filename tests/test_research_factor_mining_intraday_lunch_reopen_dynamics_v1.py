from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_intraday_lunch_reopen_dynamics_v1 as lunch
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=10),
        pd.Timedelta(hours=10, minutes=30),
        pd.Timedelta(hours=11),
        pd.Timedelta(hours=11, minutes=10),
        pd.Timedelta(hours=11, minutes=15),
        pd.Timedelta(hours=11, minutes=20),
        pd.Timedelta(hours=11, minutes=25),
        pd.Timedelta(hours=11, minutes=30),
        pd.Timedelta(hours=13),
        pd.Timedelta(hours=13, minutes=5),
        pd.Timedelta(hours=13, minutes=10),
        pd.Timedelta(hours=13, minutes=15),
        pd.Timedelta(hours=13, minutes=20),
        pd.Timedelta(hours=13, minutes=30),
        pd.Timedelta(hours=14),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _complete_panel(*, code: str = BOND) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    count = 100.0
    amount = 20_000.0
    for sequence, clock in enumerate(_times()):
        if sequence <= 8:
            price = 100.0 + 0.08 * sequence + 0.04 * np.sin(sequence * 1.1)
        else:
            # A noon jump followed by an adjustment path that has enough
            # nonzero moves to exercise the three price-resolution signals.
            price = 101.25 - 0.06 * (sequence - 9) + 0.035 * np.sin(sequence * 0.9)
        count += float(2 + ((sequence * 5 + 1) % 6))
        amount += float((2 + ((sequence * 5 + 1) % 6)) * (650 + 11 * (sequence % 5)))
        ask_depth = float(29 + ((sequence * 7 + 4) % 23))
        bid_depth = float(32 + ((sequence * 11 + 1) % 29))
        mid = price + 0.012 * np.sin(sequence * 0.57)
        spread = 0.09 + 0.006 * (sequence % 4)
        rows.append(
            {
                "dt": DT,
                "code": code,
                "seq": sequence,
                "trade_time": DT.normalize() + clock,
                "last": float(price),
                "num_trades": count,
                "amount": amount,
                "ask_price1": float(mid + spread / 2.0),
                "bid_price1": float(mid - spread / 2.0),
                "ask_volume1": ask_depth,
                "bid_volume1": bid_depth,
            }
        )
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=lunch.KERNEL_NAME, params={"signal": entry.signal})
        for entry in lunch.factor_mining_catalog()
    ]


def _build(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    return build_factor_frame(_complete_panel() if panel is None else panel, _specs())


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[[0]].copy()
    prior["seq"] = 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 10_000.0
    prior["num_trades"] = 1_000_000.0
    prior["amount"] = 1_000_000_000.0

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["last"] = 20_000.0
    preopen["num_trades"] = 2_000_000.0
    preopen["amount"] = 2_000_000_000.0

    late = frame.iloc[[-1]].copy()
    late["seq"] = 3_000
    late["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    late["last"] = 30_000.0
    late["num_trades"] = 3_000_000.0
    late["amount"] = 3_000_000_000.0

    out = pd.concat([prior, frame, preopen, late], ignore_index=True).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_lunch_reopen_catalogue_is_distinct_and_panel_only() -> None:
    entries = lunch.factor_mining_catalog()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "lunch_price_discontinuity_resolution": 3,
        "lunch_trade_resumption_recovery": 3,
        "lunch_book_rebuild_dynamics": 3,
    }
    assert set(lunch.FORMULAS) == {entry.signal for entry in entries}
    assert "lunch_reopen_return" not in {entry.signal for entry in entries}
    assert FactorRegistry.get(lunch.KERNEL_NAME) is lunch.FactorMiningIntradayLunchReopenDynamicsV1
    assert lunch.FactorMiningIntradayLunchReopenDynamicsV1.requires_stock_panel is False
    assert lunch.FactorMiningIntradayLunchReopenDynamicsV1.requires_bond_stock_map is False
    assert lunch.FactorMiningIntradayLunchReopenDynamicsV1.daily_requirements() == []


def test_lunch_reopen_builds_full_catalogue_without_inf_from_panel_only() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in lunch.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert int(frame.notna().sum(axis=1).iloc[0]) == 9
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_lunch_reopen_ignores_relabelled_prior_preopen_and_after_cutoff_rows() -> None:
    baseline = _build()
    contaminated = _build(_with_out_of_contract_rows(_complete_panel()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_lunch_reopen_fails_closed_on_missing_pre_or_post_window() -> None:
    missing_pre = _complete_panel().reset_index()
    pre_clocks = missing_pre["trade_time"].dt.time
    missing_pre = missing_pre.loc[
        ~((pre_clocks >= pd.Timestamp("11:10").time()) & (pre_clocks <= pd.Timestamp("11:30").time()))
    ]
    pre_panel = missing_pre.set_index(["dt", "code", "seq"])
    pre_panel.attrs["__build_day__"] = DT.date().isoformat()
    assert _build(pre_panel).isna().all(axis=None)

    missing_post = _complete_panel().reset_index()
    missing_post = missing_post.loc[missing_post["trade_time"].dt.time < pd.Timestamp("13:00").time()]
    post_panel = missing_post.set_index(["dt", "code", "seq"])
    post_panel.attrs["__build_day__"] = DT.date().isoformat()
    assert _build(post_panel).isna().all(axis=None)


@pytest.mark.parametrize("column", ["num_trades", "amount"])
def test_lunch_reopen_fails_closed_on_counter_reset(column: str) -> None:
    reset = _complete_panel().reset_index()
    reset.loc[11, column] = reset.loc[10, column] - 1.0
    panel = reset.set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    assert _build(panel).isna().all(axis=None)


@pytest.mark.parametrize("column, value", [("last", 0.0), ("ask_price1", 0.0), ("bid_volume1", -1.0)])
def test_lunch_reopen_fails_closed_on_invalid_required_input(column: str, value: float) -> None:
    bad = _complete_panel().reset_index()
    bad.loc[10, column] = value
    panel = bad.set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    assert _build(panel).isna().all(axis=None)


def test_lunch_reopen_requires_fields_and_rejects_unknown_signal() -> None:
    missing = _complete_panel().drop(columns=["amount"])
    missing.attrs["__build_day__"] = DT.date().isoformat()
    with pytest.raises(KeyError, match="amount"):
        lunch.FactorMiningIntradayLunchReopenDynamicsV1().compute(
            FactorComputeContext(panel=missing, params={"signal": lunch._PRICE_RESOLUTION_SIGNALS[0]})
        )
    with pytest.raises(KeyError, match="unknown signal"):
        lunch.FactorMiningIntradayLunchReopenDynamicsV1().compute(
            FactorComputeContext(panel=_complete_panel(), params={"signal": "lrd_unknown"})
        )
