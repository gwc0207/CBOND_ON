from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_quote_geometry_microprice_v1 as geometry
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(21)) + tuple(
    f"127{index:03d}.SZ" for index in range(21)
)
_CLOCKS = ("09:30", "09:45", "10:15", "10:30", "11:00", "11:30", "13:00", "13:30", "13:45", "14:00", "14:15", "14:29")


def _panel(codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    levels = np.arange(5, dtype="float64")
    steps = np.arange(4, dtype="float64") - 1.5
    for code_index, code in enumerate(codes):
        base = 99.0 + 0.91 * code_index
        for seq, clock in enumerate(_CLOCKS):
            mid_log_move = (
                0.0022 * np.sin(0.29 * code_index)
                + 0.0005 * seq
                + 0.0011 * np.sin(0.51 * seq + 0.17 * code_index)
            )
            midpoint = base * float(np.exp(mid_log_move))
            touch_half = 0.00075 + 0.00010 * (code_index % 5) + 0.000025 * seq
            bid_curvature = 0.26 * np.sin(0.23 * code_index + 0.37 * seq)
            ask_curvature = 0.24 * np.cos(0.19 * code_index - 0.31 * seq)
            bid_gaps = (0.00030 + 0.000018 * (code_index % 7)) * (1.0 + bid_curvature * steps)
            ask_gaps = (0.00034 + 0.000021 * (code_index % 6)) * (1.0 + ask_curvature * steps)
            bid_offsets = np.concatenate(([0.0], np.cumsum(bid_gaps)))
            ask_offsets = np.concatenate(([0.0], np.cumsum(ask_gaps)))
            bid_prices = midpoint * np.exp(-touch_half - bid_offsets)
            ask_prices = midpoint * np.exp(touch_half + ask_offsets)

            bid_shape = 0.31 * np.sin(0.21 * code_index + 0.48 * seq)
            ask_shape = 0.29 * np.cos(0.18 * code_index - 0.41 * seq)
            bid_depth = (900.0 + 33.0 * code_index) * np.exp(bid_shape * levels + 0.05 * np.sin(levels + seq))
            ask_depth = (870.0 + 29.0 * code_index) * np.exp(-ask_shape * levels + 0.04 * np.cos(levels + seq))
            last_offset = 0.00055 * np.sin(0.39 * code_index + 0.71 * seq)
            row: dict[str, object] = {
                "dt": SCORE,
                "code": code,
                "seq": seq,
                "trade_time": pd.Timestamp(f"{SCORE.date()} {clock}:00"),
                "last": midpoint * float(np.exp(last_offset)),
            }
            for level in range(1, 6):
                row[f"bid_price{level}"] = float(bid_prices[level - 1])
                row[f"ask_price{level}"] = float(ask_prices[level - 1])
                row[f"bid_volume{level}"] = float(bid_depth[level - 1])
                row[f"ask_volume{level}"] = float(ask_depth[level - 1])
            rows.append(row)
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=geometry.KERNEL_NAME, params={"signal": entry.signal})
        for entry in geometry.quote_geometry_microprice_catalog()
    ]


def _with_after_cutoff_rows(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel.reset_index().copy()
    appended: list[dict[str, object]] = []
    for code in CODES:
        row = base.loc[(base["code"] == code) & (base["seq"] == len(_CLOCKS) - 1)].iloc[0].to_dict()
        row["seq"] = 99
        row["trade_time"] = pd.Timestamp(f"{SCORE.date()} 14:30:00")
        row["last"] = float(row["last"]) * 10.0
        for level in range(1, 6):
            row[f"ask_price{level}"] = float(row[f"ask_price{level}"]) * 5.0
            row[f"bid_price{level}"] = float(row[f"bid_price{level}"]) * 0.2
        appended.append(row)
    out = pd.concat([base, pd.DataFrame(appended)], ignore_index=True).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = SCORE.date().isoformat()
    return out


def test_catalogue_has_four_two_signal_families_and_import_only_registration() -> None:
    entries = geometry.quote_geometry_microprice_catalog()

    assert len(entries) == 8
    assert len({entry.signal for entry in entries}) == 8
    assert Counter(entry.family for entry in entries) == {
        "five_level_depth_information_geometry": 2,
        "price_ladder_spacing_curvature": 2,
        "microprice_last_execution_alignment": 2,
        "bid_ask_geometry_coupling": 2,
    }
    assert set(geometry.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(geometry.KERNEL_NAME) is geometry.FactorMiningQuoteGeometryMicropriceV1
    assert geometry.FactorMiningQuoteGeometryMicropriceV1.__name__ not in factor_operators.__all__
    assert geometry.FactorMiningQuoteGeometryMicropriceV1.requires_stock_panel is False
    assert geometry.FactorMiningQuoteGeometryMicropriceV1.requires_bond_stock_map is False
    assert geometry.FactorMiningQuoteGeometryMicropriceV1.daily_requirements() == []


def test_all_eight_signals_build_without_inf_or_constants() -> None:
    frame = build_factor_frame(_panel(), _specs())

    assert frame.columns.tolist() == [entry.signal for entry in geometry.quote_geometry_microprice_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) == len(CODES)).all()
    assert (frame.nunique(dropna=True) > 1).all()


def test_after_cutoff_rows_are_excluded_from_every_signal() -> None:
    baseline = build_factor_frame(_panel(), _specs())
    after_cutoff = build_factor_frame(_with_after_cutoff_rows(_panel()), _specs())

    pd.testing.assert_frame_equal(after_cutoff, baseline, check_exact=True)


def test_crossed_book_and_duplicate_clock_fail_closed_for_affected_code() -> None:
    crossed = _panel().reset_index().copy()
    crossed_mask = (crossed["code"] == CODES[0]) & (crossed["seq"] == 5)
    original_bid = crossed.loc[crossed_mask, "bid_price1"].copy()
    crossed.loc[crossed_mask, "ask_price1"] = original_bid.to_numpy(dtype="float64") * 0.99
    crossed = crossed.set_index(["dt", "code", "seq"])
    crossed.attrs["__build_day__"] = SCORE.date().isoformat()
    frame = build_factor_frame(crossed, _specs())
    assert frame.loc[(SCORE, CODES[0])].isna().all()
    assert frame.loc[(SCORE, CODES[1])].notna().all()

    duplicate = _panel().reset_index().copy()
    duplicate_mask = (duplicate["code"] == CODES[2]) & (duplicate["seq"] == 6)
    duplicate.loc[duplicate_mask, "trade_time"] = pd.Timestamp(f"{SCORE.date()} 13:30:00")
    duplicate = duplicate.set_index(["dt", "code", "seq"])
    duplicate.attrs["__build_day__"] = SCORE.date().isoformat()
    duplicate_frame = build_factor_frame(duplicate, _specs())
    assert duplicate_frame.loc[(SCORE, CODES[2])].isna().all()
    assert duplicate_frame.loc[(SCORE, CODES[3])].notna().all()


def test_missing_field_and_unknown_signal_are_explicit() -> None:
    without_l5 = _panel().drop(columns=["ask_volume5"])
    with pytest.raises(KeyError, match="ask_volume5"):
        geometry.FactorMiningQuoteGeometryMicropriceV1().compute(
            FactorComputeContext(panel=without_l5, params={"signal": "qgeo_depth_entropy_asymmetry_mean"})
        )
    with pytest.raises(KeyError, match="unknown signal"):
        geometry.FactorMiningQuoteGeometryMicropriceV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "not_a_geometry_signal"})
        )
