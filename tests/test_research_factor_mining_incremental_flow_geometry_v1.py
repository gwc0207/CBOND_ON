from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_incremental_flow_geometry_v1 as flow_geometry
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"


def _times() -> list[pd.Timestamp]:
    morning = [DT.normalize() + pd.Timedelta(hours=9, minutes=30 + 5 * index) for index in range(13)]
    afternoon = [DT.normalize() + pd.Timedelta(hours=13, minutes=5 * index) for index in range(13)]
    return [*morning, *afternoon]


def _panel(*, code: str = BOND) -> pd.DataFrame:
    amount = 1_000.0
    volume = 100.0
    trades = 10.0
    last = 100.0
    quote_mid = 100.0
    quote_spread = 0.02
    imbalance = 0.0
    quote_bid_volume = 300.0
    quote_ask_volume = 310.0
    rows: list[dict[str, object]] = []
    for seq, trade_time in enumerate(_times()):
        if seq:
            direction = 1.0 if seq % 5 in {1, 2, 4} else -1.0
            last += direction * (0.006 + 0.003 * (seq % 4))
            delta_volume = 80.0 + 30.0 * ((seq * 3) % 7)
            delta_trades = 1.0 + float(seq % 5)
            volume += delta_volume
            trades += delta_trades
            amount += delta_volume * last * (1.0 + 0.06 * (seq % 4))
        if seq % 3:
            quote_mid = last + (0.004 if seq % 2 else -0.003)
            quote_spread = 0.014 + 0.004 * (seq % 4)
            imbalance = -0.65 + 0.22 * (seq % 7)
            quote_bid_volume = 300.0 + 23.0 * (seq % 8) + 40.0 * max(imbalance, 0.0)
            quote_ask_volume = 310.0 + 19.0 * ((seq + 2) % 8) + 40.0 * max(-imbalance, 0.0)
        bid1 = quote_mid - quote_spread / 2.0
        ask1 = quote_mid + quote_spread / 2.0
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": seq,
            "trade_time": trade_time,
            "last": last,
            "amount": amount,
            "volume": volume,
            "num_trades": trades,
        }
        for level in range(1, 6):
            row[f"bid_price{level}"] = bid1 - 0.002 * (level - 1)
            row[f"ask_price{level}"] = ask1 + 0.002 * (level - 1)
            row[f"bid_volume{level}"] = quote_bid_volume * (1.0 + 0.12 * (level - 1))
            row[f"ask_volume{level}"] = quote_ask_volume * (1.0 + 0.10 * (level - 1))
        rows.append(row)
    frame = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    frame.attrs["__build_day__"] = DT.date().isoformat()
    return frame


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=flow_geometry.KERNEL_NAME,
            params={"signal": entry.signal, "family": entry.family},
        )
        for entry in flow_geometry.factor_mining_catalog()
    ]


def _contaminated(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    extras: list[pd.DataFrame] = []
    for sequence, stamp in (
        (10_000, DT.normalize() - pd.Timedelta(minutes=1)),
        (10_001, DT.normalize() + pd.Timedelta(hours=9, minutes=20)),
        (10_002, DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)),
        (10_003, DT.normalize() + pd.Timedelta(hours=14, minutes=30)),
    ):
        row = frame.iloc[[-1]].copy()
        row["seq"] = sequence
        row["trade_time"] = stamp
        row["last"] = 1.0
        row["amount"] = 99_999_999.0
        row["volume"] = 99_999_999.0
        row["num_trades"] = 99_999_999.0
        extras.append(row)
    output = pd.concat([frame, *extras], ignore_index=True).set_index(["dt", "code", "seq"])
    output.attrs["__build_day__"] = DT.date().isoformat()
    return output


def test_incremental_flow_catalogue_registration_and_context_contract() -> None:
    entries = flow_geometry.factor_mining_catalog()

    assert len(entries) == 80
    assert len({entry.signal for entry in entries}) == 80
    assert Counter(entry.family for entry in entries) == {
        family: 5 for family, _signals, _hypothesis in flow_geometry._FAMILY_SPECS
    }
    assert FactorRegistry.get(flow_geometry.KERNEL_NAME) is flow_geometry.FactorMiningIncrementalFlowGeometryV1
    assert set(flow_geometry.FORMULAS) == {entry.signal for entry in entries}
    assert not flow_geometry.FactorMiningIncrementalFlowGeometryV1.requires_stock_panel
    assert not flow_geometry.FactorMiningIncrementalFlowGeometryV1.requires_bond_stock_map
    assert flow_geometry.FactorMiningIncrementalFlowGeometryV1.daily_requirements() == []


def test_incremental_flow_builds_finite_family_candidates_without_inf() -> None:
    output = build_factor_frame(_panel(), _specs())

    assert output.index.names == ["dt", "code"]
    assert output.index.tolist() == [(DT, BOND)]
    assert output.columns.tolist() == [entry.signal for entry in flow_geometry.factor_mining_catalog()]
    assert int(output.notna().sum(axis=1).iloc[0]) == 80
    assert not np.isinf(output.to_numpy(dtype="float64")).any()
    assert output.loc[(DT, BOND), "ifg_amount_increment_entropy"] > 0.0
    assert output.loc[(DT, BOND), "ifg_trade_size_log_dispersion"] > 0.0
    assert output.loc[(DT, BOND), "ifg_amount_burst_frequency"] > 0.0


def test_incremental_flow_uses_counter_deltas_not_counter_levels() -> None:
    baseline = build_factor_frame(_panel(), _specs())
    shifted = _panel().reset_index()
    for column, offset in (("amount", 9_000_000.0), ("volume", 90_000.0), ("num_trades", 9_000.0)):
        shifted[column] = shifted[column] + offset
    shifted_panel = shifted.set_index(["dt", "code", "seq"])
    shifted_panel.attrs["__build_day__"] = DT.date().isoformat()

    output = build_factor_frame(shifted_panel, _specs())

    pd.testing.assert_frame_equal(output, baseline, check_exact=False, rtol=1e-11, atol=1e-13)


def test_incremental_flow_ignores_preopen_and_after_cutoff_contamination() -> None:
    baseline = build_factor_frame(_panel(), _specs())
    output = build_factor_frame(_contaminated(_panel()), _specs())

    pd.testing.assert_frame_equal(output, baseline, check_exact=True)


def test_incremental_flow_counter_reset_or_missing_core_field_fails_closed() -> None:
    reset = _panel().reset_index()
    reset.loc[reset.index[12], "amount"] = 1.0
    reset_panel = reset.set_index(["dt", "code", "seq"])
    reset_panel.attrs["__build_day__"] = DT.date().isoformat()
    reset_output = build_factor_frame(reset_panel, _specs())

    missing = _panel().drop(columns=["volume"])
    missing.attrs["__build_day__"] = DT.date().isoformat()
    missing_output = build_factor_frame(missing, _specs())

    assert reset_output.isna().all(axis=None)
    assert missing_output.isna().all(axis=None)
    assert not np.isinf(reset_output.to_numpy(dtype="float64")).any()


def test_incremental_flow_rejects_unknown_signal() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        flow_geometry.FactorMiningIncrementalFlowGeometryV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "ifg_unknown"})
        )
