from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.operators.t1430_amount_accel_depth_delta_v2 import (
    T1430AmountAccelDepthDeltaV2,
)


DT = pd.Timestamp("2026-07-30 14:30:00")


def _rows(code: str, *, amounts: list[float], times: list[str]) -> list[dict[str, object]]:
    return [
        {
            "dt": DT,
            "code": code,
            "seq": seq,
            "trade_time": pd.Timestamp(f"2026-07-30 {tick}"),
            "amount": amount,
            "bid_volume1": 3.0,
            "ask_volume1": 1.0,
        }
        for seq, (amount, tick) in enumerate(zip(amounts, times, strict=True))
    ]


def _valid_rows(code: str = "110001.SH") -> list[dict[str, object]]:
    # Early flow is 40, recent flow is 100, and L1 imbalance is 0.5.
    return _rows(
        code,
        amounts=[0.0, 0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0],
        times=[
            "14:18:30",
            "14:19:00",
            "14:20:00",
            "14:21:00",
            "14:22:00",
            "14:23:00",
            "14:24:00",
            "14:25:00",
            "14:26:00",
            "14:27:00",
            "14:28:00",
            "14:29:00",
        ],
    )


def _panel(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index(["dt", "code", "seq"])


def _compute(rows: list[dict[str, object]]) -> pd.Series:
    return T1430AmountAccelDepthDeltaV2().compute(FactorComputeContext(panel=_panel(rows)))


def test_delta_amount_factor_is_registered_and_panel_only() -> None:
    assert FactorRegistry.get("t1430_amount_accel_depth_delta_v2") is T1430AmountAccelDepthDeltaV2
    assert T1430AmountAccelDepthDeltaV2.requires_stock_panel is False
    assert T1430AmountAccelDepthDeltaV2.requires_bond_stock_map is False
    assert T1430AmountAccelDepthDeltaV2.daily_requirements() == []


def test_delta_amount_factor_uses_true_incremental_flow_and_excludes_post_cutoff_tick() -> None:
    baseline = _compute(_valid_rows())
    late = _valid_rows()
    late.append(
        {
            "dt": DT,
            "code": "110001.SH",
            "seq": 99,
            "trade_time": pd.Timestamp("2026-07-30 14:30:00"),
            "amount": 1.0,
            "bid_volume1": 999.0,
            "ask_volume1": 1.0,
        }
    )
    with_late = _compute(late)

    expected = math.log1p(1.25)
    assert baseline.loc[(DT, "110001.SH")] == pytest.approx(expected)
    assert with_late.loc[(DT, "110001.SH")] == pytest.approx(expected)


def test_delta_amount_factor_never_differences_across_trade_dates() -> None:
    prior_dt = pd.Timestamp("2026-07-29 14:30:00")
    prior = [
        {
            "dt": prior_dt,
            "code": "110001.SH",
            "seq": 0,
            "trade_time": pd.Timestamp("2026-07-29 14:28:00"),
            "amount": 10000.0,
            "bid_volume1": 3.0,
            "ask_volume1": 1.0,
        },
        {
            "dt": prior_dt,
            "code": "110001.SH",
            "seq": 1,
            "trade_time": pd.Timestamp("2026-07-29 14:29:00"),
            "amount": 10050.0,
            "bid_volume1": 3.0,
            "ask_volume1": 1.0,
        },
    ]

    actual = _compute(prior + _valid_rows())

    assert actual.loc[(DT, "110001.SH")] == pytest.approx(math.log1p(1.25))


def test_delta_amount_factor_ignores_earlier_session_counter_transitions() -> None:
    earlier_session = [
        {
            "dt": DT,
            "code": "110001.SH",
            "seq": -2,
            "trade_time": pd.Timestamp("2026-07-30 09:14:59"),
            "amount": 1000000.0,
            "bid_volume1": 3.0,
            "ask_volume1": 1.0,
        },
        {
            "dt": DT,
            "code": "110001.SH",
            "seq": -1,
            "trade_time": pd.Timestamp("2026-07-30 09:15:01"),
            "amount": 0.0,
            "bid_volume1": 3.0,
            "ask_volume1": 1.0,
        },
    ]

    actual = _compute(earlier_session + _valid_rows())

    assert actual.loc[(DT, "110001.SH")] == pytest.approx(math.log1p(1.25))


def test_delta_amount_factor_fails_fast_on_cumulative_reset_or_negative_amount() -> None:
    reset = _valid_rows()
    reset[-1]["amount"] = 10.0
    with pytest.raises(ValueError, match="correction exceeds tolerance"):
        _compute(reset)

    negative = _valid_rows()
    negative[4]["amount"] = -1.0
    with pytest.raises(ValueError, match="negative cumulative amount"):
        _compute(negative)


def test_delta_amount_factor_preserves_a_bounded_vendor_correction_as_signed_flow() -> None:
    corrected = _valid_rows()
    # A half-mill correction at a 100-unit cumulative amount is within both the
    # configured absolute and relative guard; it is not clipped to zero.
    corrected[-2]["amount"] = 99.9995

    actual = _compute(corrected)

    assert actual.loc[(DT, "110001.SH")] == pytest.approx(math.log1p(1.25))


def test_delta_amount_factor_keeps_missing_seed_or_zero_early_flow_explicitly_missing() -> None:
    no_seed = _valid_rows("110002.SH")
    no_seed = [row for row in no_seed if row["trade_time"] > pd.Timestamp("2026-07-30 14:19:00")]

    zero_early = _valid_rows("110003.SH")
    for row in zero_early:
        if row["trade_time"] <= pd.Timestamp("2026-07-30 14:24:00"):
            row["amount"] = 0.0
        elif row["trade_time"] > pd.Timestamp("2026-07-30 14:24:00"):
            row["amount"] = float((row["seq"] - 6) * 20)

    actual = _compute(_valid_rows("110001.SH") + no_seed + zero_early)

    assert np.isfinite(actual.loc[(DT, "110001.SH")])
    assert math.isnan(actual.loc[(DT, "110002.SH")])
    assert math.isnan(actual.loc[(DT, "110003.SH")])


def test_delta_amount_factor_keeps_a_wholly_unusable_day_as_all_nan() -> None:
    # A market-wide early-window snapshot gap is a missing factor date, not a
    # factor-computation exception. Its code index must be retained so the
    # downstream strict-calendar OOS audit can see the gap explicitly.
    no_seed = [
        row
        for row in _valid_rows("110004.SH")
        if row["trade_time"] > pd.Timestamp("2026-07-30 14:19:00")
    ]

    actual = _compute(no_seed)

    assert actual.name == "t1430_amount_accel_depth_delta_v2"
    assert actual.index.tolist() == [(DT, "110004.SH")]
    assert actual.isna().all()
