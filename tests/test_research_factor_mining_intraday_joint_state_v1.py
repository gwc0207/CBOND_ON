from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_intraday_joint_state_v1 as joint
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
STOCK = "600001.SH"
OTHER_STOCK = "600002.SH"


def _times() -> list[pd.Timedelta]:
    return [
        *[pd.Timedelta(hours=9, minutes=30 + 5 * index) for index in range(6)],
        *[pd.Timedelta(hours=10, minutes=15 * index) for index in range(6)],
        *[pd.Timedelta(hours=13, minutes=5 * index) for index in range(6)],
        pd.Timedelta(hours=13, minutes=30),
        pd.Timedelta(hours=13, minutes=40),
        pd.Timedelta(hours=13, minutes=50),
        pd.Timedelta(hours=14),
        pd.Timedelta(hours=14, minutes=10),
        pd.Timedelta(hours=14, minutes=20),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _complete_panel(*, code: str, base: float, stock_shape: bool = False) -> pd.DataFrame:
    clocks = _times()
    indices = np.arange(len(clocks), dtype="float64")
    phase = np.array([0.0] * 6 + [0.5] * 6 + [-0.25] * 6 + [0.7] * 7, dtype="float64")
    wobble = 0.005 * np.sin(indices * (0.91 if stock_shape else 1.17))
    drift = (0.0008 if stock_shape else 0.0011) * indices
    prices = base * (1.0 + drift + phase * (0.006 if stock_shape else 0.007) + wobble)
    rows: list[dict[str, object]] = []
    for sequence, (clock, price) in enumerate(zip(clocks, prices, strict=True)):
        mid = float(price * (1.0 + 0.0007 * np.sin(sequence * (1.7 if stock_shape else 1.3))))
        spread = float(base * (0.00065 + 0.00007 * ((sequence * 3 + (1 if stock_shape else 0)) % 5)))
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": sequence,
            "trade_time": DT.normalize() + clock,
            "pre_close": float(base * (0.987 if stock_shape else 0.984)),
            "open": float(base * (1.006 if stock_shape else 1.009)),
            "last": float(price),
        }
        ask_volume = float(31 + ((sequence * 5 + (2 if stock_shape else 0)) % 17))
        bid_volume = float(34 + ((sequence * 7 + (0 if stock_shape else 3)) % 19))
        row.update(
            {
                "ask_price1": mid + spread / 2.0,
                "bid_price1": mid - spread / 2.0,
                "ask_volume1": ask_volume,
                "bid_volume1": bid_volume,
            }
        )
        rows.append(row)
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _daily_data(*, mapping_stock: str = STOCK, include_score_and_future: bool = False) -> dict[str, pd.DataFrame]:
    prior = DT.normalize() - pd.offsets.BDay(1)
    price_rows: list[dict[str, object]] = [
        {
            "trade_date": prior,
            "code": "110001",
            "exchange_code": "SH",
            "close_price": 100.0,
        }
    ]
    base_rows: list[dict[str, object]] = [
        {
            "trade_date": prior,
            "code": "110001",
            "exchange_code": "SH",
            "stock_code": mapping_stock,
        }
    ]
    if include_score_and_future:
        for day in (DT.normalize(), DT.normalize() + pd.Timedelta(days=1)):
            price_rows.append(
                {
                    "trade_date": day,
                    "code": "110001",
                    "exchange_code": "SH",
                    "close_price": 1000.0,
                }
            )
            base_rows.append(
                {
                    "trade_date": day,
                    "code": "110001",
                    "exchange_code": "SH",
                    "stock_code": OTHER_STOCK,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=joint.KERNEL_NAME, params={"signal": entry.signal})
        for entry in joint.factor_mining_catalog()
    ]


def _build(*, daily_data: dict[str, pd.DataFrame] | None = None, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    bond = _complete_panel(code=BOND, base=100.0) if panel is None else panel
    stock = _complete_panel(code=STOCK, base=10.0, stock_shape=True)
    other_stock = _complete_panel(code=OTHER_STOCK, base=15.0, stock_shape=False)
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
    prior = frame.iloc[:2].copy()
    prior["seq"] = prior["seq"] + 1000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 10_000.0

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 2000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["last"] = 20_000.0

    after_1429 = frame.iloc[[-1]].copy()
    after_1429["seq"] = 3000
    after_1429["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_1429["last"] = 30_000.0

    at_1430 = frame.iloc[[-1]].copy()
    at_1430["seq"] = 3001
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    at_1430["last"] = 40_000.0

    contaminated = pd.concat([prior, frame, preopen, after_1429, at_1430], ignore_index=True).set_index(["dt", "code", "seq"])
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    return contaminated


def test_joint_catalogue_has_seven_distinct_families_and_registered_kernel() -> None:
    entries = joint.factor_mining_catalog()

    assert len(entries) == 42
    assert len({entry.signal for entry in entries}) == 42
    assert Counter(entry.family for entry in entries) == {
        "joint_opening_gap_transmission": 6,
        "joint_phase_vector_coherence": 6,
        "joint_quote_trade_channel_transmission": 6,
        "joint_stock_shock_bond_book_gate": 6,
        "joint_cross_book_price_elasticity": 6,
        "joint_cross_book_state_synchrony": 6,
        "joint_tail_cojump_containment": 6,
    }
    assert set(joint.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(joint.KERNEL_NAME) is joint.FactorMiningIntradayJointStateV1
    assert joint.FactorMiningIntradayJointStateV1.requires_stock_panel is True
    assert joint.FactorMiningIntradayJointStateV1.requires_bond_stock_map is False
    assert [(item.source, item.columns, item.lookback_days) for item in joint.FactorMiningIntradayJointStateV1.daily_requirements()] == [
        ("market_cbond.daily_price", ("exchange_code", "close_price"), 10),
        ("market_cbond.daily_base", ("exchange_code", "stock_code"), 10),
    ]


def test_joint_kernel_builds_full_catalogue_without_inf_and_uses_tminus1_mapping() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in joint.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert int(frame.notna().sum(axis=1).iloc[0]) == 42


def test_joint_kernel_ignores_score_day_future_daily_rows_and_conflicting_context_map() -> None:
    baseline = _build(daily_data=_daily_data())
    contaminated = _build(daily_data=_daily_data(include_score_and_future=True))
    tminus1_other_stock = _build(daily_data=_daily_data(mapping_stock=OTHER_STOCK))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)
    # ``_build`` always supplies OTHER_STOCK through ctx.bond_stock_map.  Only
    # changing the certified T-1 daily-base mapping may change this factor.
    assert baseline.loc[(DT, BOND), "joint_tail_signed_cojump"] != pytest.approx(
        tminus1_other_stock.loc[(DT, BOND), "joint_tail_signed_cojump"]
    )


def test_joint_kernel_fails_closed_when_only_a_stale_base_mapping_exists() -> None:
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


def test_joint_kernel_ignores_relabelled_prior_preopen_and_after_1429_rows() -> None:
    baseline = _build()
    contaminated = _build(panel=_with_out_of_contract_rows(_complete_panel(code=BOND, base=100.0)))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_joint_kernel_requires_all_joint_book_fields_without_fallback() -> None:
    panel = _complete_panel(code=BOND, base=100.0).drop(columns=["ask_volume1"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    with pytest.raises(KeyError, match="ask_volume1"):
        joint.FactorMiningIntradayJointStateV1().compute(
            FactorComputeContext(
                panel=panel,
                stock_panel=_complete_panel(code=STOCK, base=10.0, stock_shape=True),
                daily_data=_daily_data(),
                params={"signal": "joint_tail_signed_cojump"},
            )
        )


def test_joint_kernel_rejects_unknown_signal_explicitly() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        joint.FactorMiningIntradayJointStateV1().compute(
            FactorComputeContext(panel=_complete_panel(code=BOND, base=100.0), params={"signal": "joint_unknown"})
        )
