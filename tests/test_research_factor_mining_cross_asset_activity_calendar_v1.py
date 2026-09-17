from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_cross_asset_activity_calendar_v1 as calendar
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
STOCK = "600001.SH"
OTHER_STOCK = "600002.SH"


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=9, minutes=35),
        pd.Timedelta(hours=9, minutes=40),
        pd.Timedelta(hours=9, minutes=45),
        pd.Timedelta(hours=9, minutes=50),
        pd.Timedelta(hours=10, minutes=5),
        pd.Timedelta(hours=10, minutes=20),
        pd.Timedelta(hours=10, minutes=45),
        pd.Timedelta(hours=11, minutes=10),
        pd.Timedelta(hours=13, minutes=0),
        pd.Timedelta(hours=13, minutes=20),
        pd.Timedelta(hours=13, minutes=45),
        pd.Timedelta(hours=14, minutes=5),
        pd.Timedelta(hours=14, minutes=25),
    ]


def _activity_shape(shape: int) -> list[float]:
    if shape == 0:
        return [1, 1, 1, 2, 1, 2, 1, 2, 1, 4, 4, 6, 8, 10]
    if shape == 1:
        return [1, 10, 8, 7, 5, 4, 4, 3, 2, 2, 1, 1, 1, 1]
    return [5, 1, 7, 2, 6, 3, 8, 2, 5, 2, 9, 1, 6, 2]


def _panel(*, code: str, shape: int) -> pd.DataFrame:
    cumulative = 100.0
    rows: list[dict[str, object]] = []
    for seq, (clock, increment) in enumerate(zip(_times(), _activity_shape(shape), strict=True)):
        cumulative += increment
        rows.append(
            {
                "dt": DT,
                "code": code,
                "seq": seq,
                "trade_time": DT.normalize() + clock,
                "num_trades": cumulative,
            }
        )
    out = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _daily_data(*, mapping_stock: str = STOCK, include_score_and_future: bool = False) -> dict[str, pd.DataFrame]:
    prior = DT.normalize() - pd.offsets.BDay(1)
    price_rows: list[dict[str, object]] = [
        {"trade_date": prior, "code": "110001", "exchange_code": "SH", "close_price": 100.0}
    ]
    base_rows: list[dict[str, object]] = [
        {"trade_date": prior, "code": "110001", "exchange_code": "SH", "stock_code": mapping_stock}
    ]
    if include_score_and_future:
        for day in (DT.normalize(), DT.normalize() + pd.Timedelta(days=1)):
            price_rows.append({"trade_date": day, "code": "110001", "exchange_code": "SH", "close_price": 999.0})
            base_rows.append({"trade_date": day, "code": "110001", "exchange_code": "SH", "stock_code": OTHER_STOCK})
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=calendar.KERNEL_NAME, params={"signal": entry.signal})
        for entry in calendar.factor_mining_catalog()
    ]


def _build(
    *,
    daily_data: dict[str, pd.DataFrame] | None = None,
    bond_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    bond = _panel(code=BOND, shape=0) if bond_panel is None else bond_panel
    stock = _panel(code=STOCK, shape=1)
    other_stock = _panel(code=OTHER_STOCK, shape=2)
    stock_panel = pd.concat([stock.reset_index(), other_stock.reset_index()], ignore_index=True).set_index(["dt", "code", "seq"])
    stock_panel.attrs["__build_day__"] = DT.date().isoformat()
    return build_factor_frame(
        bond,
        _specs(),
        stock_panel=stock_panel,
        bond_stock_map=pd.DataFrame({"code": [BOND], "stock_code": [OTHER_STOCK]}),
        daily_data=daily_data or _daily_data(),
    )


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[[0]].copy()
    prior["seq"] = 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["num_trades"] = 1_000_000.0

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["num_trades"] = 2_000_000.0

    after_1429 = frame.iloc[[-1]].copy()
    after_1429["seq"] = 3_000
    after_1429["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_1429["num_trades"] = 3_000_000.0

    at_1430 = frame.iloc[[-1]].copy()
    at_1430["seq"] = 3_001
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    at_1430["num_trades"] = 4_000_000.0

    out = pd.concat([prior, frame, preopen, after_1429, at_1430], ignore_index=True).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_calendar_catalogue_registration_and_context_contract() -> None:
    entries = calendar.factor_mining_catalog()

    assert len(entries) == 6
    assert len({entry.signal for entry in entries}) == 6
    assert Counter(entry.family for entry in entries) == {
        "cross_asset_activity_calendar_geometry": 3,
        "cross_asset_activity_calendar_burst_asymmetry": 3,
    }
    assert set(calendar.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(calendar.KERNEL_NAME) is calendar.FactorMiningCrossAssetActivityCalendarV1
    assert calendar.FactorMiningCrossAssetActivityCalendarV1.requires_stock_panel is True
    assert calendar.FactorMiningCrossAssetActivityCalendarV1.requires_bond_stock_map is False
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in calendar.FactorMiningCrossAssetActivityCalendarV1.daily_requirements()
    ] == [
        ("market_cbond.daily_price", ("exchange_code", "close_price"), 10),
        ("market_cbond.daily_base", ("exchange_code", "stock_code"), 10),
    ]


def test_calendar_kernel_builds_six_finite_nonprice_signals_from_tminus1_mapping() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in calendar.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert int(frame.notna().sum(axis=1).iloc[0]) == 6
    assert frame.loc[(DT, BOND), "xca_activity_calendar_center_gap"] > 0.0
    assert frame.loc[(DT, BOND), "xca_activity_calendar_late_tail_share_gap"] > 0.0


def test_calendar_uses_certified_tminus1_mapping_and_ignores_context_score_and_future_rows() -> None:
    baseline = _build(daily_data=_daily_data())
    contaminated = _build(daily_data=_daily_data(include_score_and_future=True))
    tminus1_other_stock = _build(daily_data=_daily_data(mapping_stock=OTHER_STOCK))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)
    assert baseline.loc[(DT, BOND), "xca_activity_calendar_center_gap"] != pytest.approx(
        tminus1_other_stock.loc[(DT, BOND), "xca_activity_calendar_center_gap"]
    )


def test_calendar_fails_closed_for_stale_mapping_without_inf() -> None:
    prior = DT.normalize() - pd.offsets.BDay(1)
    stale = prior - pd.offsets.BDay(1)
    daily_data = {
        "market_cbond.daily_price": pd.DataFrame(
            [{"trade_date": prior, "code": "110001", "exchange_code": "SH", "close_price": 100.0}]
        ),
        "market_cbond.daily_base": pd.DataFrame(
            [{"trade_date": stale, "code": "110001", "exchange_code": "SH", "stock_code": STOCK}]
        ),
    }

    frame = _build(daily_data=daily_data)

    assert frame.index.tolist() == [(DT, BOND)]
    assert frame.isna().all(axis=None)
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_calendar_ignores_nonphysical_preopen_and_after_cutoff_rows() -> None:
    baseline = _build()
    contaminated = _build(bond_panel=_with_out_of_contract_rows(_panel(code=BOND, shape=0)))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_calendar_counter_reset_becomes_nan_not_infinite() -> None:
    panel = _panel(code=BOND, shape=0).reset_index()
    panel.loc[panel.index[-1], "num_trades"] = 1.0
    reset_panel = panel.set_index(["dt", "code", "seq"])
    reset_panel.attrs["__build_day__"] = DT.date().isoformat()

    frame = _build(bond_panel=reset_panel)

    assert frame.isna().all(axis=None)
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_calendar_rejects_unknown_signal() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        calendar.FactorMiningCrossAssetActivityCalendarV1().compute(
            FactorComputeContext(
                panel=_panel(code=BOND, shape=0),
                params={"signal": "xca_unknown"},
            )
        )
