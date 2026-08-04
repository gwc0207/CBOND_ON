from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_expansion_intraday_v1 as expansion
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
    volume_steps = np.array(
        [20, 35, 29, 45, 32, 50, 38, 62, 47, 41, 58, 46, 70, 52, 65, 49, 74, 61, 83, 57],
        dtype="float64",
    )
    amount_steps = volume_steps * prices * np.array(
        [1.01, 0.99, 1.02, 1.00, 1.03, 0.98, 1.02, 1.01, 0.97, 1.04] * 2,
        dtype="float64",
    )
    trade_steps = np.array(
        [3, 5, 4, 7, 5, 8, 6, 10, 7, 6, 9, 7, 11, 8, 10, 7, 12, 9, 13, 8],
        dtype="float64",
    )
    volume = np.cumsum(volume_steps)
    amount = np.cumsum(amount_steps)
    num_trades = np.cumsum(trade_steps)
    rows: list[dict[str, object]] = []
    for sequence, (clock, price) in enumerate(zip(_times(), prices, strict=True)):
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": sequence,
            "trade_time": DT.normalize() + clock,
            "pre_close": 100.0,
            "last": float(price),
            "volume": float(volume[sequence]),
            "amount": float(amount[sequence]),
            "num_trades": float(num_trades[sequence]),
            "high_limited": 120.0,
            "low_limited": 80.0,
            "iopv": float(100.50 + 0.055 * sequence + 0.07 * np.sin(sequence)),
        }
        for level in range(1, 6):
            ask_distance = 0.009 * level + 0.001 * ((sequence + level) % 4)
            bid_distance = 0.008 * level + 0.001 * ((2 * sequence + level) % 3)
            row[f"ask_price{level}"] = float(price + ask_distance)
            row[f"bid_price{level}"] = float(price - bid_distance)
            row[f"ask_volume{level}"] = float(22 + 4 * level + ((3 * sequence + level) % 9))
            row[f"bid_volume{level}"] = float(24 + 5 * level + ((2 * sequence + 2 * level) % 11))
        rows.append(row)
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=expansion.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in expansion.factor_mining_expansion_intraday_catalog()
    ]


def _with_physical_prior_and_late_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[:5].copy()
    prior["seq"] = prior["seq"] + 100
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 10_000.0
    prior["high_limited"] = 12_000.0
    prior["low_limited"] = 1.0

    late = frame.iloc[[-1]].copy()
    late["seq"] = 999
    late["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=31)
    late["last"] = 20_000.0
    late["high_limited"] = 24_000.0
    late["low_limited"] = 1.0

    contaminated = pd.concat([prior, frame, late], ignore_index=True).set_index(["dt", "code", "seq"])
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    return contaminated


def test_expansion_catalogue_has_ten_families_sixty_signals_and_registered_kernel() -> None:
    entries = expansion.factor_mining_expansion_intraday_catalog()

    assert len(entries) == 60
    assert len({entry.signal for entry in entries}) == 60
    assert Counter(entry.family for entry in entries) == {
        "open_gap_absorption": 6,
        "clock_time_rotation": 6,
        "multiscale_noise_variance": 6,
        "execution_price_dispersion": 6,
        "book_depth_curve_cliff": 6,
        "liquidity_supply_depletion": 6,
        "bond_limit_pressure": 6,
        "iopv_basis_convergence": 6,
        "quote_trade_stickiness": 6,
        "flow_return_lead_lag": 6,
    }
    assert {entry.kernel for entry in entries} == {expansion.KERNEL_NAME}
    assert FactorRegistry.get(expansion.KERNEL_NAME) is expansion.FactorMiningIntradayExpansionV1
    assert all(entry.hypothesis for entry in entries)


def test_expansion_kernel_builds_full_family_catalogue_to_dt_code_series_contract() -> None:
    frame = build_factor_frame(_complete_panel(), _specs())

    expected_columns = [entry.signal for entry in expansion.factor_mining_expansion_intraday_catalog()]
    assert frame.columns.tolist() == expected_columns
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    # The synthetic panel has sufficient observations, nonzero limits, and
    # valid IOPV/book fields for every candidate definition.
    assert int(frame.notna().sum(axis=1).iloc[0]) == 60


def test_expansion_ignores_relabelled_prior_rows_and_post_1430_rows() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs())
    contaminated = build_factor_frame(
        _with_physical_prior_and_late_rows(_complete_panel()),
        _specs(),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_expansion_requires_signal_specific_fields_without_fallback() -> None:
    panel = _complete_panel().drop(columns=["iopv"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    with pytest.raises(KeyError, match="iopv"):
        expansion.FactorMiningIntradayExpansionV1().compute(
            FactorComputeContext(
                panel=panel,
                params={"signal": "exp_iopv_basis_last"},
            )
        )


def test_zero_bond_limit_is_missing_not_an_at_limit_observation() -> None:
    panel = _complete_panel().copy()
    panel["high_limited"] = 0.0
    panel.attrs["__build_day__"] = DT.date().isoformat()

    out = expansion.FactorMiningIntradayExpansionV1().compute(
        FactorComputeContext(
            panel=panel,
            params={"signal": "exp_limit_up_distance"},
        )
    )

    assert out.index.tolist() == [(DT, BOND)]
    assert np.isnan(out.iloc[0])


def test_factor_spec_instance_name_keeps_expansion_kernel_dispatch_explicit() -> None:
    spec = FactorSpec(
        name="research_alias_for_basis",
        factor=expansion.KERNEL_NAME,
        params={"signal": "exp_iopv_basis_last"},
    )

    frame = build_factor_frame(_complete_panel(), [spec])

    assert frame.columns.tolist() == ["research_alias_for_basis"]
    assert np.isfinite(frame.iloc[0, 0])
