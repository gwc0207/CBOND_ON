from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_intraday_cross_section_residual_v1 as residual
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"{110000 + index:06d}.SH" for index in range(48))


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=9, minutes=34),
        pd.Timedelta(hours=9, minutes=41),
        pd.Timedelta(hours=9, minutes=50),
        pd.Timedelta(hours=10),
        pd.Timedelta(hours=10, minutes=12),
        pd.Timedelta(hours=10, minutes=25),
        pd.Timedelta(hours=10, minutes=38),
        pd.Timedelta(hours=10, minutes=52),
        pd.Timedelta(hours=11, minutes=8),
        pd.Timedelta(hours=11, minutes=21),
        pd.Timedelta(hours=11, minutes=30),
        pd.Timedelta(hours=13),
        pd.Timedelta(hours=13, minutes=8),
        pd.Timedelta(hours=13, minutes=18),
        pd.Timedelta(hours=13, minutes=31),
        pd.Timedelta(hours=13, minutes=44),
        pd.Timedelta(hours=13, minutes=57),
        pd.Timedelta(hours=14, minutes=8),
        pd.Timedelta(hours=14, minutes=17),
        pd.Timedelta(hours=14, minutes=24),
        pd.Timedelta(hours=14, minutes=28),
        pd.Timedelta(hours=14, minutes=29),
        pd.Timedelta(hours=14, minutes=30),
    ]


def _panel(codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clocks = _times()
    for code_index, code in enumerate(codes):
        pre_close = 92.0 + 1.55 * code_index
        open_gap = 0.010 * np.sin(code_index * 0.41)
        trend = 0.00017 * ((code_index % 11) - 5)
        curvature = 0.000035 * ((code_index % 7) - 3)
        volume_cum = 0.0
        amount_cum = 0.0
        trades_cum = 0.0
        for sequence, clock in enumerate(clocks):
            phase = float(sequence) / float(len(clocks) - 1)
            late = max(0.0, phase - 0.60)
            log_price = (
                np.log(pre_close)
                + open_gap
                + trend * sequence
                + curvature * (sequence - 11) ** 2
                + 0.0032 * np.sin(0.77 * sequence + 0.19 * code_index)
                + 0.0075 * np.sin(0.29 * code_index) * late
            )
            price = float(np.exp(log_price))
            volume_step = float(45.0 + 1.9 * code_index + 6.0 * (sequence % 5) + 9.0 * np.sin(0.51 * sequence + 0.14 * code_index))
            trades_step = float(3.0 + (code_index % 9) + (sequence % 6) + (1 if sequence >= 15 else 0))
            volume_step = max(1.0, volume_step)
            trades_step = max(1.0, trades_step)
            amount_step = volume_step * price * (0.88 + 0.04 * (code_index % 6) + 0.02 * np.cos(sequence + code_index))
            volume_cum += volume_step
            amount_cum += amount_step
            trades_cum += trades_step

            mid = price * (1.0 + 0.00022 * np.sin(0.63 * sequence + 0.31 * code_index))
            spread = price * (0.00032 + 0.000025 * (code_index % 8) + 0.00003 * (sequence % 4))
            row: dict[str, object] = {
                "dt": DT,
                "code": code,
                "seq": sequence,
                "trade_time": DT.normalize() + clock,
                "last": price,
                "pre_close": pre_close,
                "volume": volume_cum,
                "amount": amount_cum,
                "num_trades": trades_cum,
            }
            for level in range(1, 6):
                distance = spread * (0.5 + 0.52 * (level - 1))
                row[f"ask_price{level}"] = mid + distance
                row[f"bid_price{level}"] = mid - distance
                row[f"ask_volume{level}"] = float(15.0 + 2.1 * level + 0.9 * code_index + ((sequence * (level + 1)) % 9))
                row[f"bid_volume{level}"] = float(17.0 + 1.8 * level + 1.2 * (code_index % 10) + ((2 * sequence * level + level) % 11))
            rows.append(row)
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=residual.KERNEL_NAME, params={"signal": entry.signal})
        for entry in residual.factor_mining_intraday_cross_section_residual_catalog()
    ]


def _out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    source = frame.loc[(frame["code"] == CODES[0])].iloc[[5]].copy()
    prior = source.copy()
    prior["seq"] = 10_001
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 10_000.0
    prior["amount"] = 99_000_000.0

    preopen = source.copy()
    preopen["seq"] = 10_002
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["last"] = 20_000.0

    lunch = source.copy()
    lunch["seq"] = 10_003
    lunch["trade_time"] = DT.normalize() + pd.Timedelta(hours=12)
    lunch["last"] = 30_000.0

    late = source.copy()
    late["seq"] = 10_004
    late["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=31)
    late["last"] = 40_000.0

    contaminated = pd.concat([frame, prior, preopen, lunch, late], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    return contaminated


def _with_single_counter_decrease(
    panel: pd.DataFrame,
    *,
    code: str,
    column: str,
    decrease: float,
    sequence: int = 12,
) -> pd.DataFrame:
    """Make one real negative cumulative increment without changing its clock."""

    frame = panel.reset_index().copy()
    current = frame.index[(frame["code"] == code) & (frame["seq"] == sequence)]
    previous = frame.index[(frame["code"] == code) & (frame["seq"] == sequence - 1)]
    assert len(current) == 1 and len(previous) == 1
    frame.loc[current[0], column] = float(frame.loc[previous[0], column]) - float(decrease)
    out = frame.set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_catalogue_has_six_families_thirty_six_signals_and_import_only_kernel() -> None:
    entries = residual.factor_mining_intraday_cross_section_residual_catalog()

    assert len(entries) == 36
    assert len({entry.signal for entry in entries}) == 36
    assert Counter(entry.family for entry in entries) == {
        "cs_price_action_abnormality": 6,
        "cs_flow_activity_anomaly": 6,
        "cs_book_state_anomaly": 6,
        "cs_phase_rotation_anomaly": 6,
        "cs_impact_response_anomaly": 6,
        "cs_path_risk_anomaly": 6,
    }
    assert {entry.kernel for entry in entries} == {residual.KERNEL_NAME}
    assert FactorRegistry.get(residual.KERNEL_NAME) is residual.FactorMiningIntradayCrossSectionResidualV1
    assert residual.FactorMiningIntradayCrossSectionResidualV1.__name__ not in factor_operators.__all__
    assert residual.FactorMiningIntradayCrossSectionResidualV1.requires_stock_panel is False
    assert residual.FactorMiningIntradayCrossSectionResidualV1.requires_bond_stock_map is False
    assert residual.FactorMiningIntradayCrossSectionResidualV1.daily_requirements() == []


def test_cross_section_kernel_builds_all_residuals_without_inf() -> None:
    frame = build_factor_frame(_panel(), _specs())

    assert frame.columns.tolist() == [
        entry.signal for entry in residual.factor_mining_intraday_cross_section_residual_catalog()
    ]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= len(CODES) - 2).all()
    assert (frame.abs().sum(axis=0) > 0.0).all()


def test_physical_prior_preopen_lunch_and_post_cutoff_rows_are_ignored() -> None:
    baseline = build_factor_frame(_panel(), _specs())
    contaminated = build_factor_frame(_out_of_contract_rows(_panel()), _specs())

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_duplicate_clock_path_fails_closed_for_that_bond_without_cross_section_fallback() -> None:
    baseline = _panel().reset_index()
    duplicate = baseline.loc[(baseline["code"] == CODES[0])].iloc[[8]].copy()
    duplicate["seq"] = 20_001
    # Keep the exact same physical timestamp; the kernel must not pick one.
    broken = pd.concat([baseline, duplicate], ignore_index=True).set_index(["dt", "code", "seq"])
    broken.attrs["__build_day__"] = DT.date().isoformat()

    observed = build_factor_frame(broken, _specs())

    assert observed.loc[(DT, CODES[0])].isna().all()
    assert observed.loc[(DT, CODES[1])].notna().any()
    assert not np.isinf(observed.to_numpy(dtype="float64")).any()


def test_bounded_amount_vendor_correction_is_signed_not_clipped_and_keeps_all_signals() -> None:
    code = CODES[0]
    panel = _with_single_counter_decrease(_panel(), code=code, column="amount", decrease=0.05)
    score = residual._score_day_frame(FactorComputeContext(panel=panel))
    group = score.loc[score["code"] == code]

    increments = residual._counter_increments(
        group,
        "amount",
        allow_bounded_amount_correction=True,
    )
    assert increments is not None
    assert (increments < 0.0).any()

    observed = build_factor_frame(panel, _specs())

    assert observed.loc[(DT, code)].notna().all()
    assert not np.isinf(observed.to_numpy(dtype="float64")).any()


def test_material_amount_reset_only_suppresses_amount_dependent_signals_for_that_bond() -> None:
    code = CODES[0]
    panel = _with_single_counter_decrease(_panel(), code=code, column="amount", decrease=101.0)
    observed = build_factor_frame(panel, _specs())
    amount_dependent = {
        "csr_flow_amount_residual",
        "csr_flow_trade_notional_residual",
        "csr_flow_concentration_residual",
        "csr_flow_tail_share_residual",
        "csr_flow_phase_acceleration_residual",
        "csr_impact_amount_return_correlation_residual",
        "csr_impact_price_per_amount_residual",
        "csr_impact_trade_size_return_residual",
    }
    independent = [column for column in observed.columns if column not in amount_dependent]

    assert observed.loc[(DT, code), sorted(amount_dependent)].isna().all()
    assert observed.loc[(DT, code), independent].notna().all()
    assert observed.loc[(DT, CODES[1])].notna().all()


def test_volume_counter_decrease_only_suppresses_its_impact_signal_for_that_bond() -> None:
    code = CODES[0]
    panel = _with_single_counter_decrease(_panel(), code=code, column="volume", decrease=2.0)
    observed = build_factor_frame(panel, _specs())
    volume_signal = "csr_impact_volume_return_correlation_residual"
    independent = [column for column in observed.columns if column != volume_signal]

    assert pd.isna(observed.loc[(DT, code), volume_signal])
    assert observed.loc[(DT, code), independent].notna().all()
    assert observed.loc[(DT, CODES[1])].notna().all()


def test_missing_required_field_is_explicit_and_does_not_substitute_another_field() -> None:
    panel = _panel().drop(columns=["amount"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    with pytest.raises(KeyError, match="amount"):
        residual.FactorMiningIntradayCrossSectionResidualV1().compute(
            FactorComputeContext(
                panel=panel,
                params={"signal": "csr_price_terminal_return_residual"},
            )
        )


def test_ridge_residual_requires_full_finite_common_cross_section() -> None:
    source = pd.DataFrame(
        {
            "target": np.linspace(-1.0, 1.0, residual._MIN_CROSS_SECTION - 1),
            "x1": np.linspace(2.0, 3.0, residual._MIN_CROSS_SECTION - 1),
            "x2": np.linspace(5.0, 4.0, residual._MIN_CROSS_SECTION - 1),
        }
    )

    observed = residual._rank_ridge_residual(source, target="target", covariates=("x1", "x2"))

    assert observed.isna().all()
