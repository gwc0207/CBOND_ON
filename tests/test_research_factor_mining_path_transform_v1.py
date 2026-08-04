from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_path_transform_v1 as transforms
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
        pd.Timedelta(hours=11, minutes=15),
        pd.Timedelta(hours=11, minutes=29),
        pd.Timedelta(hours=13, minutes=1),
        pd.Timedelta(hours=13, minutes=20),
        pd.Timedelta(hours=13, minutes=35),
        pd.Timedelta(hours=13, minutes=45),
        pd.Timedelta(hours=14),
        pd.Timedelta(hours=14, minutes=10),
        pd.Timedelta(hours=14, minutes=20),
        pd.Timedelta(hours=14, minutes=25),
        pd.Timedelta(hours=14, minutes=29),
        pd.Timedelta(hours=14, minutes=30),
    ]


def _complete_panel(*, code: str = BOND) -> pd.DataFrame:
    prices = np.array(
        [
            101.20,
            102.10,
            101.35,
            101.80,
            100.95,
            101.40,
            101.05,
            101.75,
            102.35,
            101.90,
            101.25,
            102.05,
            102.65,
            102.15,
            101.55,
            102.30,
            102.75,
            102.20,
            102.95,
            102.60,
        ],
        dtype="float64",
    )
    rows: list[dict[str, object]] = []
    for sequence, (clock, price) in enumerate(zip(_times(), prices, strict=True)):
        rows.append(
            {
                "dt": DT,
                "code": code,
                "seq": sequence,
                "trade_time": DT.normalize() + clock,
                "pre_close": 100.0,
                "last": float(price),
            }
        )
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=transforms.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in transforms.factor_mining_path_transform_catalog()
    ]


def _with_physical_prior_and_late_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[:5].copy()
    prior["seq"] = prior["seq"] + 100
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 10_000.0
    prior["pre_close"] = 9_000.0

    late = frame.iloc[[-1]].copy()
    late["seq"] = 999
    late["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=31)
    late["last"] = 20_000.0
    late["pre_close"] = 18_000.0

    contaminated = pd.concat([prior, frame, late], ignore_index=True).set_index(["dt", "code", "seq"])
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    return contaminated


def test_path_transform_catalogue_has_three_families_twenty_four_signals_and_registered_kernel() -> None:
    entries = transforms.factor_mining_path_transform_catalog()

    assert len(entries) == 24
    assert len({entry.signal for entry in entries}) == 24
    assert Counter(entry.family for entry in entries) == {
        "intraday_phase_transition": 8,
        "extremum_sequence": 8,
        "alpha_path_transform_port": 8,
    }
    assert {entry.kernel for entry in entries} == {transforms.KERNEL_NAME}
    assert FactorRegistry.get(transforms.KERNEL_NAME) is transforms.FactorMiningPathTransformV1
    assert all(entry.hypothesis for entry in entries)


def test_path_transform_kernel_builds_all_signals_to_dt_code_contract() -> None:
    frame = build_factor_frame(_complete_panel(), _specs())

    expected_columns = [entry.signal for entry in transforms.factor_mining_path_transform_catalog()]
    assert frame.columns.tolist() == expected_columns
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert int(frame.notna().sum(axis=1).iloc[0]) == 24


def test_path_transform_ignores_relabelled_prior_rows_and_post_1430_rows() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    contaminated = build_factor_frame(
        _with_physical_prior_and_late_rows(_complete_panel()),
        _specs(),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_path_transform_requires_last_without_fallback() -> None:
    panel = _complete_panel().drop(columns=["last"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    with pytest.raises(KeyError, match="last"):
        transforms.FactorMiningPathTransformV1().compute(
            FactorComputeContext(
                panel=panel,
                params={"signal": "path_extreme_high_low_order"},
            )
        )


def test_path_transform_unknown_signal_is_explicit_error() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        transforms.FactorMiningPathTransformV1().compute(
            FactorComputeContext(
                panel=_complete_panel(),
                params={"signal": "path_not_a_registered_signal"},
            )
        )
