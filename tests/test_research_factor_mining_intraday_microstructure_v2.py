from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_intraday_microstructure_v2 as micro
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=9, minutes=35),
        pd.Timedelta(hours=9, minutes=45),
        pd.Timedelta(hours=10),
        pd.Timedelta(hours=10, minutes=15),
        pd.Timedelta(hours=10, minutes=25),
        pd.Timedelta(hours=10, minutes=35),
        pd.Timedelta(hours=10, minutes=50),
        pd.Timedelta(hours=11, minutes=5),
        pd.Timedelta(hours=11, minutes=20),
        pd.Timedelta(hours=11, minutes=30),
        pd.Timedelta(hours=13),
        pd.Timedelta(hours=13, minutes=10),
        pd.Timedelta(hours=13, minutes=20),
        pd.Timedelta(hours=13, minutes=30),
        pd.Timedelta(hours=13, minutes=40),
        pd.Timedelta(hours=13, minutes=50),
        pd.Timedelta(hours=14),
        pd.Timedelta(hours=14, minutes=8),
        pd.Timedelta(hours=14, minutes=15),
        pd.Timedelta(hours=14, minutes=20),
        pd.Timedelta(hours=14, minutes=25),
        pd.Timedelta(hours=14, minutes=29),
        pd.Timedelta(hours=14, minutes=30),
    ]


def _complete_panel(*, code: str = BOND) -> pd.DataFrame:
    prices = np.array(
        [
            101.20,
            101.65,
            101.10,
            101.70,
            100.90,
            101.45,
            101.00,
            101.60,
            102.20,
            101.75,
            102.10,
            101.20,
            101.85,
            102.50,
            102.00,
            102.60,
            102.20,
            102.85,
            102.35,
            103.00,
            102.55,
            103.10,
            102.70,
            103.00,
        ],
        dtype="float64",
    )
    trade_steps = np.array(
        [3, 0, 5, 0, 7, 2, 0, 9, 1, 0, 6, 3, 0, 12, 2, 0, 7, 1, 0, 14, 3, 0, 8, 4],
        dtype="float64",
    )
    trade_counts = np.cumsum(trade_steps)
    quote_offsets = np.array(
        [
            0.12,
            -0.15,
            0.10,
            -0.11,
            0.14,
            -0.07,
            0.16,
            -0.12,
            0.05,
            -0.13,
            0.11,
            -0.08,
            0.14,
            -0.10,
            0.07,
            -0.14,
            0.12,
            -0.06,
            0.15,
            -0.09,
            0.10,
            -0.13,
            0.08,
            -0.11,
        ],
        dtype="float64",
    )
    rows: list[dict[str, object]] = []
    for sequence, (clock, price) in enumerate(zip(_times(), prices, strict=True)):
        mid = float(price + quote_offsets[sequence])
        spread = float(0.065 + 0.008 * (sequence % 3))
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": sequence,
            "trade_time": DT.normalize() + clock,
            "last": float(price),
            "num_trades": float(trade_counts[sequence]),
        }
        for level in range(1, 6):
            row[f"ask_price{level}"] = mid + spread / 2.0 + 0.008 * (level - 1)
            row[f"bid_price{level}"] = mid - spread / 2.0 - 0.007 * (level - 1)
            row[f"ask_volume{level}"] = float(18 + 3 * level + ((sequence * (level + 2)) % 11))
            row[f"bid_volume{level}"] = float(21 + 4 * level + ((2 * sequence * level + level) % 13))
        rows.append(row)
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=micro.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in micro.factor_mining_intraday_microstructure_v2_catalog()
    ]


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[:4].copy()
    prior["seq"] = prior["seq"] + 100
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 10_000.0
    prior["num_trades"] = 1_000_000.0

    late = frame.iloc[[-1]].copy()
    late["seq"] = 999
    late["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=31)
    late["last"] = 20_000.0
    late["num_trades"] = 2_000_000.0

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 1000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["last"] = 30_000.0

    lunch = frame.iloc[[0]].copy()
    lunch["seq"] = 1001
    lunch["trade_time"] = DT.normalize() + pd.Timedelta(hours=12)
    lunch["last"] = 40_000.0

    contaminated = pd.concat([prior, frame, late, preopen, lunch], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    return contaminated


def test_microstructure_catalogue_has_three_families_twenty_four_signals_and_registered_kernel() -> None:
    entries = micro.factor_mining_intraday_microstructure_v2_catalog()

    assert len(entries) == 24
    assert len({entry.signal for entry in entries}) == 24
    assert Counter(entry.family for entry in entries) == {
        "quote_trade_path_asynchrony": 8,
        "event_clock_price_discovery": 8,
        "depth_centroid_relocation": 8,
    }
    assert {entry.kernel for entry in entries} == {micro.KERNEL_NAME}
    assert FactorRegistry.get(micro.KERNEL_NAME) is micro.FactorMiningIntradayMicrostructureV2
    assert all(entry.hypothesis for entry in entries)


def test_microstructure_kernel_builds_full_catalogue_to_dt_code_contract_without_inf() -> None:
    frame = build_factor_frame(_complete_panel(), _specs())

    expected_columns = [entry.signal for entry in micro.factor_mining_intraday_microstructure_v2_catalog()]
    assert frame.columns.tolist() == expected_columns
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert int(frame.notna().sum(axis=1).iloc[0]) == 24


def test_microstructure_ignores_relabelled_prior_post_cutoff_preopen_and_lunch_rows() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    contaminated = build_factor_frame(_with_out_of_contract_rows(_complete_panel()), _specs())

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_microstructure_discards_crossed_quote_ticks_for_quote_and_depth_families() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    frame = _complete_panel().reset_index()
    crossed = frame.iloc[[7]].copy()
    crossed["seq"] = 2000
    crossed["trade_time"] = DT.normalize() + pd.Timedelta(hours=10, minutes=42)
    crossed["ask_price1"] = 1.0
    crossed["bid_price1"] = 999.0
    crossed["ask_price2"] = 1.1
    crossed["bid_price2"] = 998.0
    crossed["ask_price3"] = 1.2
    crossed["bid_price3"] = 997.0
    crossed["ask_price4"] = 1.3
    crossed["bid_price4"] = 996.0
    crossed["ask_price5"] = 1.4
    crossed["bid_price5"] = 995.0
    contaminated = pd.concat([frame, crossed], ignore_index=True).set_index(["dt", "code", "seq"])
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    observed = build_factor_frame(contaminated, _specs())

    book_columns = [
        entry.signal
        for entry in micro.factor_mining_intraday_microstructure_v2_catalog()
        if entry.family in {"quote_trade_path_asynchrony", "depth_centroid_relocation"}
    ]
    pd.testing.assert_frame_equal(observed[book_columns], baseline[book_columns], check_exact=True)


def test_microstructure_nonmonotonic_trade_count_fails_closed_for_event_clock_family() -> None:
    panel = _complete_panel().copy()
    panel.iloc[10, panel.columns.get_loc("num_trades")] = panel.iloc[9, panel.columns.get_loc("num_trades")] - 1.0
    panel.attrs["__build_day__"] = DT.date().isoformat()
    event_specs = [
        spec
        for spec, entry in zip(_specs(), micro.factor_mining_intraday_microstructure_v2_catalog(), strict=True)
        if entry.family == "event_clock_price_discovery"
    ]

    frame = build_factor_frame(panel, event_specs)

    assert frame.index.tolist() == [(DT, BOND)]
    assert frame.isna().all(axis=None)


def test_microstructure_requires_family_specific_fields_without_fallback() -> None:
    panel = _complete_panel().drop(columns=["num_trades"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    with pytest.raises(KeyError, match="num_trades"):
        micro.FactorMiningIntradayMicrostructureV2().compute(
            FactorComputeContext(
                panel=panel,
                params={"signal": "micro_event_intensity_dispersion"},
            )
        )


def test_microstructure_unknown_signal_is_explicit_error() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        micro.FactorMiningIntradayMicrostructureV2().compute(
            FactorComputeContext(
                panel=_complete_panel(),
                params={"signal": "micro_not_a_registered_signal"},
            )
        )
