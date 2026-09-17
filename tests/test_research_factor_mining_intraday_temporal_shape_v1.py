from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_intraday_temporal_shape_v1 as temporal_shape
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=9, minutes=35),
        pd.Timedelta(hours=9, minutes=40),
        pd.Timedelta(hours=9, minutes=45),
        pd.Timedelta(hours=9, minutes=50),
        pd.Timedelta(hours=9, minutes=55),
        pd.Timedelta(hours=10),
        pd.Timedelta(hours=10, minutes=5),
        pd.Timedelta(hours=10, minutes=10),
        pd.Timedelta(hours=10, minutes=15),
        pd.Timedelta(hours=10, minutes=20),
        pd.Timedelta(hours=10, minutes=25),
        pd.Timedelta(hours=10, minutes=30),
        pd.Timedelta(hours=10, minutes=35),
        pd.Timedelta(hours=10, minutes=40),
        pd.Timedelta(hours=10, minutes=45),
        pd.Timedelta(hours=10, minutes=50),
        pd.Timedelta(hours=10, minutes=55),
        pd.Timedelta(hours=11),
        pd.Timedelta(hours=11, minutes=5),
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
        pd.Timedelta(hours=13, minutes=25),
        pd.Timedelta(hours=13, minutes=30),
        pd.Timedelta(hours=13, minutes=35),
        pd.Timedelta(hours=13, minutes=40),
        pd.Timedelta(hours=13, minutes=45),
        pd.Timedelta(hours=13, minutes=50),
        pd.Timedelta(hours=13, minutes=55),
        pd.Timedelta(hours=14),
        pd.Timedelta(hours=14, minutes=5),
        pd.Timedelta(hours=14, minutes=10),
        pd.Timedelta(hours=14, minutes=15),
        pd.Timedelta(hours=14, minutes=20),
        pd.Timedelta(hours=14, minutes=25),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _complete_panel(*, code: str = BOND) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    count = 100.0
    for sequence, clock in enumerate(_times()):
        price_step = 0.10 * np.sin(sequence * 0.77) + 0.045 * ((sequence * 3) % 5 - 2)
        price = 100.0 + 0.17 * sequence + price_step
        count += float(1 + ((sequence * 5 + 3) % 7))
        bid_depth = float(28 + ((sequence * 11 + 2) % 23))
        ask_depth = float(31 + ((sequence * 7 + 4) % 29))
        mid = price + 0.015 * np.sin(sequence * 0.41)
        spread = 0.08 + 0.005 * (sequence % 4)
        rows.append(
            {
                "dt": DT,
                "code": code,
                "seq": sequence,
                "trade_time": DT.normalize() + clock,
                "last": float(price),
                "num_trades": count,
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
        FactorSpec(name=entry.signal, factor=temporal_shape.KERNEL_NAME, params={"signal": entry.signal})
        for entry in temporal_shape.factor_mining_catalog()
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

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["last"] = 20_000.0
    preopen["num_trades"] = 2_000_000.0

    after_cutoff = frame.iloc[[-1]].copy()
    after_cutoff["seq"] = 3_000
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_cutoff["last"] = 30_000.0
    after_cutoff["num_trades"] = 3_000_000.0

    at_1430 = frame.iloc[[-1]].copy()
    at_1430["seq"] = 3_001
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    at_1430["last"] = 40_000.0
    at_1430["num_trades"] = 4_000_000.0

    out = pd.concat([prior, frame, preopen, after_cutoff, at_1430], ignore_index=True).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_temporal_shape_catalogue_registration_and_panel_only_contract() -> None:
    entries = temporal_shape.factor_mining_catalog()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "temporal_trade_variation_allocation": 3,
        "temporal_book_adjustment_allocation": 3,
        "temporal_trade_depth_profile_coupling": 3,
    }
    assert set(temporal_shape.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(temporal_shape.KERNEL_NAME) is temporal_shape.FactorMiningIntradayTemporalShapeV1
    assert temporal_shape.FactorMiningIntradayTemporalShapeV1.requires_stock_panel is False
    assert temporal_shape.FactorMiningIntradayTemporalShapeV1.requires_bond_stock_map is False
    assert temporal_shape.FactorMiningIntradayTemporalShapeV1.daily_requirements() == []


def test_temporal_shape_builds_full_catalogue_without_inf_from_panel_only() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in temporal_shape.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert int(frame.notna().sum(axis=1).iloc[0]) == 9
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_temporal_shape_is_invariant_to_raw_price_and_depth_units() -> None:
    baseline = _build()
    scaled = _complete_panel().reset_index()
    scaled["last"] *= 10.0
    scaled["ask_price1"] *= 10.0
    scaled["bid_price1"] *= 10.0
    scaled["ask_volume1"] *= 100.0
    scaled["bid_volume1"] *= 100.0
    scaled_panel = scaled.set_index(["dt", "code", "seq"])
    scaled_panel.attrs["__build_day__"] = DT.date().isoformat()

    pd.testing.assert_frame_equal(_build(scaled_panel), baseline, check_exact=False, rtol=1e-12, atol=1e-12)


def test_temporal_shape_ignores_relabelled_prior_preopen_and_after_cutoff_rows() -> None:
    baseline = _build()
    contaminated = _build(_with_out_of_contract_rows(_complete_panel()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_temporal_shape_fails_closed_on_in_session_gap_or_counter_reset() -> None:
    gap = _complete_panel().reset_index()
    gap = gap.loc[~gap["trade_time"].isin([DT.normalize() + pd.Timedelta(hours=10, minutes=5), DT.normalize() + pd.Timedelta(hours=10, minutes=10), DT.normalize() + pd.Timedelta(hours=10, minutes=15), DT.normalize() + pd.Timedelta(hours=10, minutes=20), DT.normalize() + pd.Timedelta(hours=10, minutes=25)])]
    gap_panel = gap.set_index(["dt", "code", "seq"])
    gap_panel.attrs["__build_day__"] = DT.date().isoformat()
    assert _build(gap_panel).isna().all(axis=None)

    reset = _complete_panel().reset_index()
    reset.loc[20, "num_trades"] = reset.loc[19, "num_trades"] - 1.0
    reset_panel = reset.set_index(["dt", "code", "seq"])
    reset_panel.attrs["__build_day__"] = DT.date().isoformat()
    assert _build(reset_panel).isna().all(axis=None)


@pytest.mark.parametrize("column, value", [("last", 0.0), ("ask_price1", 0.0), ("bid_volume1", -1.0)])
def test_temporal_shape_fails_closed_on_invalid_price_or_depth(column: str, value: float) -> None:
    bad = _complete_panel().reset_index()
    bad.loc[10, column] = value
    panel = bad.set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    assert _build(panel).isna().all(axis=None)


def test_temporal_shape_requires_fields_and_rejects_unknown_signal() -> None:
    missing = _complete_panel().drop(columns=["num_trades"])
    missing.attrs["__build_day__"] = DT.date().isoformat()
    with pytest.raises(KeyError, match="num_trades"):
        temporal_shape.FactorMiningIntradayTemporalShapeV1().compute(
            FactorComputeContext(panel=missing, params={"signal": temporal_shape._PRICE_ALLOCATION_SIGNALS[0]})
        )
    with pytest.raises(KeyError, match="unknown signal"):
        temporal_shape.FactorMiningIntradayTemporalShapeV1().compute(
            FactorComputeContext(panel=_complete_panel(), params={"signal": "temporal_shape_unknown"})
        )
