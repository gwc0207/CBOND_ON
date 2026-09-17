from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_cross_asset_book_geometry_v1 as geometry
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
STOCK = "600001.SH"
OTHER_STOCK = "600002.SH"


def _times() -> list[pd.Timedelta]:
    morning = [pd.Timedelta(hours=9, minutes=30 + 5 * index) for index in range(24)]
    afternoon = [pd.Timedelta(hours=13, minutes=5 * index) for index in range(18)]
    return morning + afternoon


def _panel(*, code: str, shape: int) -> pd.DataFrame:
    base_mid = 100.0 if code == BOND else 10.0
    base_spread = 0.045 if code == BOND else 0.008
    rows: list[dict[str, object]] = []
    for seq, clock in enumerate(_times()):
        mid = base_mid * (1.0 + 0.00015 * seq)
        spread = base_spread * (1.0 + 0.22 * np.sin(0.37 * seq + 0.45 * shape))
        step = spread * (0.32 + 0.04 * np.cos(0.21 * seq + shape))
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": seq,
            "trade_time": DT.normalize() + clock,
        }
        for level in range(1, 6):
            offset = (level - 1) * step
            row[f"ask_price{level}"] = mid + spread / 2.0 + offset
            row[f"bid_price{level}"] = mid - spread / 2.0 - offset
            bid_scale = 1.0 + 0.26 * np.sin(0.33 * seq + 0.72 * level + 0.41 * shape)
            ask_scale = 1.0 + 0.23 * np.cos(0.29 * seq + 0.61 * level + 0.53 * shape)
            row[f"bid_volume{level}"] = max(1.0, (105.0 + 8.0 * level) * bid_scale)
            row[f"ask_volume{level}"] = max(1.0, (95.0 + 10.0 * level) * ask_scale)
        rows.append(row)
    out = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _stock_panel() -> pd.DataFrame:
    out = pd.concat(
        [_panel(code=STOCK, shape=0).reset_index(), _panel(code=OTHER_STOCK, shape=2).reset_index()],
        ignore_index=True,
    ).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _mapping(*, stock_code: str = STOCK, trade_date: object = DT.normalize()) -> pd.DataFrame:
    return pd.DataFrame({"code": [BOND], "stock_code": [stock_code], "trade_date": [trade_date]})


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=geometry.KERNEL_NAME, params={"signal": entry.signal})
        for entry in geometry.factor_mining_catalog()
    ]


def _build(
    *,
    bond_panel: pd.DataFrame | None = None,
    stock_panel: pd.DataFrame | None = None,
    mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_factor_frame(
        bond_panel if bond_panel is not None else _panel(code=BOND, shape=1),
        _specs(),
        stock_panel=_stock_panel() if stock_panel is None else stock_panel,
        bond_stock_map=_mapping() if mapping is None else mapping,
    )


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    invalid = frame.iloc[[0]].copy()
    invalid["ask_price1"] = 1.0
    invalid["bid_price1"] = 2.0
    invalid["ask_volume1"] = 0.0
    invalid["bid_volume1"] = 0.0

    prior = invalid.copy()
    prior["seq"] = 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    preopen = invalid.copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    after_cutoff = invalid.copy()
    after_cutoff["seq"] = 3_000
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    at_1430 = invalid.copy()
    at_1430["seq"] = 3_001
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    out = pd.concat([prior, frame, preopen, after_cutoff, at_1430], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_book_geometry_catalogue_registration_and_context_contract() -> None:
    entries = geometry.factor_mining_catalog()
    requirements = infer_factor_context_requirements(_specs())

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "cross_asset_l1_liquidity_state": 3,
        "cross_asset_depth_curve_geometry": 3,
        "cross_asset_book_state_comovement": 3,
    }
    assert FactorRegistry.get(geometry.KERNEL_NAME) is geometry.FactorMiningCrossAssetBookGeometryV1
    assert set(geometry.FORMULAS) == {entry.signal for entry in entries}
    assert geometry.FactorMiningCrossAssetBookGeometryV1.requires_stock_panel is True
    assert geometry.FactorMiningCrossAssetBookGeometryV1.requires_bond_stock_map is True
    assert geometry.FactorMiningCrossAssetBookGeometryV1.daily_requirements() == []
    assert requirements.stock_panel_required is True
    assert requirements.bond_stock_map_required is True
    assert requirements.daily_required is False


def test_book_geometry_kernel_builds_nine_finite_book_only_signals() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in geometry.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert int(frame.notna().sum(axis=1).iloc[0]) == 9
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert 0.0 <= frame.loc[(DT, BOND), "cabg_joint_depth_curve_hellinger_mean"] <= 1.0
    assert -1.0 <= frame.loc[(DT, BOND), "cabg_l1_imbalance_time_correlation"] <= 1.0


def test_book_geometry_uses_context_mapping_and_rejects_explicit_future_mapping() -> None:
    baseline = _build(mapping=_mapping(stock_code=STOCK))
    other = _build(mapping=_mapping(stock_code=OTHER_STOCK))
    future = _build(mapping=_mapping(trade_date=DT.normalize() + pd.Timedelta(days=1)))

    assert baseline.loc[(DT, BOND), "cabg_joint_depth_curve_hellinger_mean"] != pytest.approx(
        other.loc[(DT, BOND), "cabg_joint_depth_curve_hellinger_mean"]
    )
    assert future.isna().all(axis=None)
    assert not np.isinf(future.to_numpy(dtype="float64")).any()


def test_book_geometry_ignores_prior_preopen_and_after_cutoff_invalid_books() -> None:
    baseline = _build()
    contaminated = _build(
        bond_panel=_with_out_of_contract_rows(_panel(code=BOND, shape=1)),
        stock_panel=_with_out_of_contract_rows(_stock_panel()),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_book_geometry_mapping_and_invalid_book_gaps_fail_closed_without_inf() -> None:
    duplicate = pd.concat([_mapping(), _mapping(stock_code=OTHER_STOCK)], ignore_index=True)
    duplicate_result = _build(mapping=duplicate)

    invalid = _panel(code=BOND, shape=1).reset_index()
    invalid.loc[invalid.index[8], "ask_price2"] = invalid.loc[invalid.index[8], "ask_price1"]
    invalid_panel = invalid.set_index(["dt", "code", "seq"])
    invalid_panel.attrs["__build_day__"] = DT.date().isoformat()
    invalid_result = _build(bond_panel=invalid_panel)

    assert duplicate_result.isna().all(axis=None)
    assert invalid_result.isna().all(axis=None)
    assert not np.isinf(duplicate_result.to_numpy(dtype="float64")).any()
    assert not np.isinf(invalid_result.to_numpy(dtype="float64")).any()


def test_book_geometry_rejects_unknown_signal() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        geometry.FactorMiningCrossAssetBookGeometryV1().compute(
            FactorComputeContext(panel=_panel(code=BOND, shape=1), params={"signal": "cabg_unknown"})
        )
