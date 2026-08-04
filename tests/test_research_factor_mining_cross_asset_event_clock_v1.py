from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_cross_asset_event_clock_v1 as event_clock
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
STOCK = "600001.SH"
OTHER_STOCK = "600002.SH"


def _times() -> list[pd.Timedelta]:
    return [
        *[pd.Timedelta(hours=9, minutes=30 + 5 * index) for index in range(10)],
        *[pd.Timedelta(hours=13, minutes=5 * index) for index in range(17)],
    ]


def _complete_panel(*, code: str, base: float, shape: int) -> pd.DataFrame:
    volume = 1_000.0
    trades = 100.0
    rows: list[dict[str, object]] = []
    for sequence, clock in enumerate(_times()):
        common_lull = sequence in {5, 15}
        if common_lull:
            trade_increment = 0.0
        elif shape == 0:
            trade_increment = float(1 + ((sequence * 2) % 4))
        elif shape == 1:
            trade_increment = float(1 + ((sequence * 3 + 1) % 5))
        else:
            trade_increment = float(1 + ((sequence + 2) % 2))
        volume_increment = 0.0 if trade_increment == 0.0 else trade_increment * float(80 + 7 * ((sequence + shape) % 5))
        volume += volume_increment
        trades += trade_increment

        midpoint = base * (1.0 + 0.0003 * sequence + 0.0001 * np.sin(sequence * (1.1 + 0.1 * shape)))
        spread = base * (0.0010 + 0.0001 * ((sequence + shape) % 3))
        direction = 1.0 if ((sequence + shape) % 4) in {0, 1} else -1.0
        location = (0.25 + 0.10 * ((sequence + 2 * shape) % 3)) * direction
        rows.append(
            {
                "dt": DT,
                "code": code,
                "seq": sequence,
                "trade_time": DT.normalize() + clock,
                "last": midpoint + location * spread / 2.0,
                "ask_price1": midpoint + spread / 2.0,
                "bid_price1": midpoint - spread / 2.0,
                "ask_volume1": float(20 + ((sequence * 3 + shape) % 11)),
                "bid_volume1": float(24 + ((sequence * 5 + shape) % 13)),
                "volume": volume,
                "num_trades": trades,
            }
        )
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


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
        FactorSpec(name=entry.signal, factor=event_clock.KERNEL_NAME, params={"signal": entry.signal})
        for entry in event_clock.factor_mining_catalog()
    ]


def _build(*, daily_data: dict[str, pd.DataFrame] | None = None, bond_panel: pd.DataFrame | None = None) -> pd.DataFrame:
    bond = _complete_panel(code=BOND, base=100.0, shape=0) if bond_panel is None else bond_panel
    stock = _complete_panel(code=STOCK, base=10.0, shape=1)
    other_stock = _complete_panel(code=OTHER_STOCK, base=15.0, shape=2)
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
    prior["volume"] = 1_000_000.0
    prior["num_trades"] = 1_000_000.0

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["volume"] = 2_000_000.0
    preopen["num_trades"] = 2_000_000.0

    after_1429 = frame.iloc[[-1]].copy()
    after_1429["seq"] = 3_000
    after_1429["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_1429["volume"] = 3_000_000.0
    after_1429["num_trades"] = 3_000_000.0

    at_1430 = frame.iloc[[-1]].copy()
    at_1430["seq"] = 3_001
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    at_1430["volume"] = 4_000_000.0
    at_1430["num_trades"] = 4_000_000.0

    out = pd.concat([prior, frame, preopen, after_1429, at_1430], ignore_index=True).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_event_clock_catalogue_registration_and_context_contract() -> None:
    entries = event_clock.factor_mining_catalog()

    assert len(entries) == 6
    assert len({entry.signal for entry in entries}) == 6
    assert Counter(entry.family for entry in entries) == {
        "cross_asset_event_clock_transmission": 3,
        "cross_asset_signed_execution_transmission": 3,
    }
    assert set(event_clock.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(event_clock.KERNEL_NAME) is event_clock.FactorMiningCrossAssetEventClockV1
    assert event_clock.FactorMiningCrossAssetEventClockV1.requires_stock_panel is True
    assert event_clock.FactorMiningCrossAssetEventClockV1.requires_bond_stock_map is False
    assert [
        (item.source, item.columns, item.lookback_days)
        for item in event_clock.FactorMiningCrossAssetEventClockV1.daily_requirements()
    ] == [
        ("market_cbond.daily_price", ("exchange_code", "close_price"), 10),
        ("market_cbond.daily_base", ("exchange_code", "stock_code"), 10),
    ]


def test_event_clock_kernel_builds_six_finite_signals_from_tminus1_mapping() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in event_clock.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert int(frame.notna().sum(axis=1).iloc[0]) == 6


def test_event_clock_uses_certified_tminus1_mapping_and_ignores_score_day_rows() -> None:
    baseline = _build(daily_data=_daily_data())
    contaminated = _build(daily_data=_daily_data(include_score_and_future=True))
    tminus1_other_stock = _build(daily_data=_daily_data(mapping_stock=OTHER_STOCK))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)
    assert baseline.loc[(DT, BOND), "xca_event_burst_coactivity"] != pytest.approx(
        tminus1_other_stock.loc[(DT, BOND), "xca_event_burst_coactivity"]
    )


def test_event_clock_fails_closed_for_stale_mapping_without_inf() -> None:
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


def test_event_clock_ignores_nonphysical_preopen_and_after_cutoff_rows() -> None:
    baseline = _build()
    contaminated = _build(bond_panel=_with_out_of_contract_rows(_complete_panel(code=BOND, base=100.0, shape=0)))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_event_clock_counter_reset_becomes_nan_not_infinite() -> None:
    panel = _complete_panel(code=BOND, base=100.0, shape=0).reset_index()
    panel.loc[panel.index[-1], "volume"] = 1.0
    panel.loc[panel.index[-1], "num_trades"] = 1.0
    reset_panel = panel.set_index(["dt", "code", "seq"])
    reset_panel.attrs["__build_day__"] = DT.date().isoformat()

    frame = _build(bond_panel=reset_panel)

    assert frame.isna().all(axis=None)
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()


def test_event_clock_rejects_unknown_signal() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        event_clock.FactorMiningCrossAssetEventClockV1().compute(
            FactorComputeContext(
                panel=_complete_panel(code=BOND, base=100.0, shape=0),
                params={"signal": "xca_unknown"},
            )
        )
