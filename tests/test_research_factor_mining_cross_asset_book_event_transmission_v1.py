from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_asset_book_event_transmission_v1 as event_transmission,
)
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND_A = "110001.SH"
BOND_B = "110002.SH"
STOCK_A = "600001.SH"
STOCK_B = "600002.SH"
OTHER_STOCK = "600003.SH"


def _times() -> list[pd.Timedelta]:
    return [pd.Timedelta(hours=9, minutes=30 + 5 * index) for index in range(20)]


def _directions(shape: int) -> np.ndarray:
    base = np.array(
        [
            0.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
        ],
        dtype="float64",
    )
    return base if shape == 0 else np.roll(base, shape)


def _panel(*, code: str, shape: int) -> pd.DataFrame:
    is_bond = code.startswith("110")
    mid = 100.0 if is_bond else 10.0
    base_spread = 0.045 if is_bond else 0.008
    tick = 0.0011 if is_bond else 0.0038
    rows: list[dict[str, object]] = []
    for seq, (clock, direction) in enumerate(
        zip(_times(), _directions(shape), strict=True)
    ):
        if seq:
            mid *= 1.0 + tick * direction
        spread = base_spread * (1.0 + 0.28 * np.sin(0.61 * seq + 0.37 * shape))
        step = spread * (0.38 + 0.03 * np.cos(0.27 * seq + 0.11 * shape))
        l1_scale = 1.0 + 0.34 * np.sin(0.73 * seq + 0.51 * shape)
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": seq,
            "trade_time": DT.normalize() + clock,
            "volume": float(seq * (97 + 11 * shape)),
            "num_trades": float(seq * (3 + shape)),
        }
        for level in range(1, 6):
            depth_scale = 1.0 + 0.13 * np.cos(0.49 * seq + 0.83 * level + 0.29 * shape)
            l1 = l1_scale if level == 1 else 1.0
            offset = (level - 1) * step
            row[f"ask_price{level}"] = mid + spread / 2.0 + offset
            row[f"bid_price{level}"] = mid - spread / 2.0 - offset
            row[f"bid_volume{level}"] = max(
                1.0, (112.0 + 9.0 * level) * depth_scale * l1
            )
            row[f"ask_volume{level}"] = max(
                1.0, (94.0 + 11.0 * level) * depth_scale / l1
            )
        rows.append(row)
    out = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _bond_panel() -> pd.DataFrame:
    out = pd.concat(
        [
            _panel(code=BOND_A, shape=1).reset_index(),
            _panel(code=BOND_B, shape=2).reset_index(),
        ],
        ignore_index=True,
    ).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _stock_panel() -> pd.DataFrame:
    out = pd.concat(
        [
            _panel(code=STOCK_A, shape=0).reset_index(),
            _panel(code=STOCK_B, shape=3).reset_index(),
            _panel(code=OTHER_STOCK, shape=4).reset_index(),
        ],
        ignore_index=True,
    ).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _mapping(
    *, trade_date: object = DT.normalize(), stock_a: str = STOCK_A
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": [BOND_A, BOND_B],
            "stock_code": [stock_a, STOCK_B],
            "trade_date": [trade_date, trade_date],
        }
    )


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=event_transmission.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in event_transmission.factor_mining_catalog()
    ]


def _build(
    *,
    bond_panel: pd.DataFrame | None = None,
    stock_panel: pd.DataFrame | None = None,
    mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_factor_frame(
        _bond_panel() if bond_panel is None else bond_panel,
        _specs(),
        stock_panel=_stock_panel() if stock_panel is None else stock_panel,
        bond_stock_map=_mapping() if mapping is None else mapping,
    )


def _single_build(
    *, bond_panel: pd.DataFrame, stock_panel: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    return build_factor_frame(
        bond_panel, _specs(), stock_panel=stock_panel, bond_stock_map=mapping
    )


def _single_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {"code": [BOND_A], "stock_code": [STOCK_A], "trade_date": [DT.normalize()]}
    )


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    invalid = frame.iloc[[0]].copy()
    invalid["ask_price1"] = 1.0
    invalid["bid_price1"] = 2.0
    invalid["volume"] = -1.0
    invalid["num_trades"] = -1.0

    prior = invalid.copy()
    prior["seq"] = 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    preopen = invalid.copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    after_cutoff = invalid.copy()
    after_cutoff["seq"] = 3_000
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(
        hours=14, minutes=29, seconds=30
    )
    at_1430 = invalid.copy()
    at_1430["seq"] = 3_001
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    future = invalid.copy()
    future["seq"] = 3_002
    future["trade_time"] = DT.normalize() + pd.Timedelta(days=1)
    out = pd.concat(
        [prior, frame, preopen, after_cutoff, at_1430, future], ignore_index=True
    ).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_catalogue_registration_and_context_contract() -> None:
    entries = event_transmission.factor_mining_catalog()
    requirements = infer_factor_context_requirements(_specs())

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "cross_asset_triggered_bond_book_recovery": 3,
        "cross_asset_trigger_direction_asymmetry": 3,
        "cross_asset_local_liquidity_state_response": 3,
    }
    assert (
        FactorRegistry.get(event_transmission.KERNEL_NAME)
        is event_transmission.FactorMiningCrossAssetBookEventTransmissionV1
    )
    assert set(event_transmission.FORMULAS) == {entry.signal for entry in entries}
    assert (
        event_transmission.FactorMiningCrossAssetBookEventTransmissionV1.requires_stock_panel
        is True
    )
    assert (
        event_transmission.FactorMiningCrossAssetBookEventTransmissionV1.requires_bond_stock_map
        is True
    )
    assert (
        event_transmission.FactorMiningCrossAssetBookEventTransmissionV1.daily_requirements()
        == []
    )
    assert requirements.stock_panel_required is True
    assert requirements.bond_stock_map_required is True
    assert requirements.daily_required is False


def test_kernel_builds_finite_nonconstant_matched_event_response_signals_without_inf() -> (
    None
):
    frame = _build()

    assert frame.columns.tolist() == [
        entry.signal for entry in event_transmission.factor_mining_catalog()
    ]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND_A), (DT, BOND_B)]
    assert frame.notna().all(axis=None)
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert int(frame.nunique(dropna=True).max()) == 2


def test_context_mapping_changes_results_and_explicit_future_mapping_fails_closed() -> (
    None
):
    baseline = _build()
    remapped = _build(mapping=_mapping(stock_a=OTHER_STOCK))
    future = _build(mapping=_mapping(trade_date=DT.normalize() + pd.Timedelta(days=1)))

    assert baseline.loc[
        (DT, BOND_A), "cabet_stock_reprice_bond_spread_recovery"
    ] != pytest.approx(
        remapped.loc[(DT, BOND_A), "cabet_stock_reprice_bond_spread_recovery"]
    )
    assert future.isna().all(axis=None)
    assert not np.isinf(future.to_numpy(dtype="float64")).any()


def test_score_day_and_future_rows_cannot_contaminate_physical_1429_output() -> None:
    baseline = _build()
    contaminated = _build(
        bond_panel=_with_out_of_contract_rows(_bond_panel()),
        stock_panel=_with_out_of_contract_rows(_stock_panel()),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_mapping_unmatched_pairs_counter_reset_and_bad_books_fail_closed_without_inf() -> (
    None
):
    bond = _panel(code=BOND_A, shape=1)
    stock = _panel(code=STOCK_A, shape=0)
    mapping = _single_mapping()

    missing_mapping = _single_build(
        bond_panel=bond, stock_panel=stock, mapping=pd.DataFrame()
    )

    unmatched_stock = stock.reset_index()
    unmatched_stock["trade_time"] = unmatched_stock["trade_time"] + pd.Timedelta(
        hours=3
    )
    unmatched_stock = unmatched_stock.set_index(["dt", "code", "seq"])
    unmatched_stock.attrs["__build_day__"] = DT.date().isoformat()
    unmatched = _single_build(
        bond_panel=bond, stock_panel=unmatched_stock, mapping=mapping
    )

    reset_bond = bond.reset_index()
    reset_bond.loc[reset_bond.index[8], "num_trades"] = (
        reset_bond.loc[reset_bond.index[7], "num_trades"] - 1.0
    )
    reset_bond = reset_bond.set_index(["dt", "code", "seq"])
    reset_bond.attrs["__build_day__"] = DT.date().isoformat()
    reset = _single_build(bond_panel=reset_bond, stock_panel=stock, mapping=mapping)

    crossed_stock = stock.reset_index()
    crossed_stock.loc[crossed_stock.index[9], "ask_price1"] = crossed_stock.loc[
        crossed_stock.index[9], "bid_price1"
    ]
    crossed_stock = crossed_stock.set_index(["dt", "code", "seq"])
    crossed_stock.attrs["__build_day__"] = DT.date().isoformat()
    crossed = _single_build(bond_panel=bond, stock_panel=crossed_stock, mapping=mapping)

    duplicate_stock = stock.reset_index()
    duplicate_row = duplicate_stock.iloc[[7]].copy()
    duplicate_row["seq"] = 1_000
    duplicate_stock = pd.concat(
        [duplicate_stock, duplicate_row], ignore_index=True
    ).set_index(["dt", "code", "seq"])
    duplicate_stock.attrs["__build_day__"] = DT.date().isoformat()
    duplicate = _single_build(
        bond_panel=bond, stock_panel=duplicate_stock, mapping=mapping
    )

    short_bond = bond.reset_index().iloc[:3].set_index(["dt", "code", "seq"])
    short_bond.attrs["__build_day__"] = DT.date().isoformat()
    short_stock = stock.reset_index().iloc[:3].set_index(["dt", "code", "seq"])
    short_stock.attrs["__build_day__"] = DT.date().isoformat()
    insufficient = _single_build(
        bond_panel=short_bond, stock_panel=short_stock, mapping=mapping
    )

    for result in (missing_mapping, unmatched, reset, crossed, duplicate, insufficient):
        assert result.isna().all(axis=None)
        assert not np.isinf(result.to_numpy(dtype="float64")).any()


def test_unknown_signal_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        event_transmission.FactorMiningCrossAssetBookEventTransmissionV1().compute(
            FactorComputeContext(
                panel=_panel(code=BOND_A, shape=1), params={"signal": "cabet_unknown"}
            )
        )
