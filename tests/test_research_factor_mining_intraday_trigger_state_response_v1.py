from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_intraday_trigger_state_response_v1 as trigger
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
BONDS = tuple(f"110{index:03d}.SH" for index in range(16))
STOCKS = tuple(f"600{index:03d}.SH" for index in range(16))


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _times() -> list[pd.Timestamp]:
    day = SCORE.normalize()
    morning = list(pd.date_range(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=11, minutes=30), freq="5min"))
    afternoon = list(pd.date_range(day + pd.Timedelta(hours=13), day + pd.Timedelta(hours=14, minutes=25), freq="5min"))
    return [*morning, *afternoon, day + pd.Timedelta(hours=14, minutes=29)]


def _panel(codes: tuple[str, ...], *, stock: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(codes):
        conv = 10.0 + 0.13 * code_index
        stock_base = conv * (1.13 + 0.005 * (code_index % 5))
        for seq, timestamp in enumerate(_times()):
            step = float(seq)
            stock_last = stock_base * np.exp(
                0.0001 * step
                + 0.0048 * np.sin(step / 2.6 + 0.22 * code_index)
                + 0.0015 * np.cos(step / 4.3 + 0.19 * code_index)
            )
            bond_last = 117.0 + 0.7 * code_index + 2.2 * np.sin(step / 2.1 + 0.31 * code_index) + 0.04 * step
            rows.append(
                {
                    "dt": SCORE,
                    "code": code,
                    "seq": seq,
                    "trade_time": timestamp,
                    "last": stock_last if stock else bond_last,
                }
            )
    return pd.DataFrame(rows).set_index(["dt", "code", "seq"])


def _mapping(*, future: bool = False, mismatch: str | None = None) -> pd.DataFrame:
    rows = []
    for bond, stock in zip(BONDS, STOCKS, strict=True):
        rows.append(
            {
                "code": bond,
                "stock_code": STOCKS[-1] if bond == mismatch else stock,
                "trade_date": SCORE.normalize() + pd.offsets.BDay(1) if future else SCORE.normalize(),
            }
        )
    return pd.DataFrame(rows)


def _daily_sources() -> dict[str, pd.DataFrame]:
    prior = SCORE.normalize() - pd.offsets.BDay(1)
    older = prior - pd.offsets.BDay(1)
    future = SCORE.normalize() + pd.offsets.BDay(1)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for index, (bond, stock) in enumerate(zip(BONDS, STOCKS, strict=True)):
        conv = 10.0 + 0.13 * index
        stock_base = conv * (1.13 + 0.005 * (index % 5))
        active = index % 2 == 0
        cumulative = 13.0 if active else 7.0
        for day in (older, prior):
            price_rows.append({"trade_date": day, "code": _bare(bond), "exchange_code": "SH", "close_price": 116.0 + index})
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(bond),
                    "exchange_code": "SH",
                    "stock_code": stock,
                    "trigger_price_revise": stock_base * (1.17 + 0.004 * np.sin(index)),
                    "cb_call_price": 130.0 + 0.75 * index,
                    "cb_put_price": 101.0 + 0.4 * index,
                    "in_trigger_process": 1.0 if active else -1.0,
                    "trigger_cum_days_revise": cumulative,
                    "trigger_reach_days_revise": 15.0,
                }
            )
        price_rows.append({"trade_date": SCORE.normalize(), "code": _bare(bond), "exchange_code": "SH", "close_price": 1_000_000.0})
        base_rows.extend(
            [
                {
                    "trade_date": SCORE.normalize(),
                    "code": _bare(bond),
                    "exchange_code": "SH",
                    "stock_code": STOCKS[-1],
                    "trigger_price_revise": 1_000_000.0,
                    "cb_call_price": 1_000_000.0,
                    "cb_put_price": 1.0,
                    "in_trigger_process": -1.0,
                    "trigger_cum_days_revise": 0.0,
                    "trigger_reach_days_revise": 1.0,
                },
                {
                    "trade_date": future,
                    "code": _bare(bond),
                    "exchange_code": "SH",
                    "stock_code": STOCKS[-1],
                    "trigger_price_revise": 2_000_000.0,
                    "cb_call_price": 2_000_000.0,
                    "cb_put_price": 1.0,
                    "in_trigger_process": -1.0,
                    "trigger_cum_days_revise": 0.0,
                    "trigger_reach_days_revise": 1.0,
                },
            ]
        )
    return {"market_cbond.daily_price": pd.DataFrame(price_rows), "market_cbond.daily_base": pd.DataFrame(base_rows)}


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=trigger.KERNEL_NAME, params={"signal": entry.signal})
        for entry in trigger.intraday_trigger_state_response_catalog()
    ]


def _build(
    *,
    bond_panel: pd.DataFrame | None = None,
    stock_panel: pd.DataFrame | None = None,
    mapping: pd.DataFrame | None = None,
    daily_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    return build_factor_frame(
        _panel(BONDS, stock=False) if bond_panel is None else bond_panel,
        _specs(),
        stock_panel=_panel(STOCKS, stock=True) if stock_panel is None else stock_panel,
        bond_stock_map=_mapping() if mapping is None else mapping,
        daily_data=_daily_sources() if daily_data is None else daily_data,
    )


def _with_post_cutoff(panel: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    extra = panel.reset_index().groupby(["dt", "code"], sort=False).tail(1).copy()
    extra["seq"] = extra["seq"].astype(int) + 10_000
    extra["trade_time"] = SCORE.normalize() + pd.Timedelta(hours=14, minutes=30)
    extra["last"] = pd.to_numeric(extra["last"], errors="coerce") * multiplier
    return pd.concat([panel.reset_index(), extra], ignore_index=True).set_index(["dt", "code", "seq"])


def test_catalogue_has_four_dynamic_families_and_research_only_registration() -> None:
    entries = trigger.intraday_trigger_state_response_catalog()

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "trigger_barrier_distance_trajectory": 3,
        "mapped_bond_trigger_approach_response": 3,
        "call_put_redemption_corridor_dynamics": 3,
        "near_completion_trigger_response": 3,
    }
    assert set(trigger.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(trigger.KERNEL_NAME) is trigger.FactorMiningIntradayTriggerStateResponseV1
    assert trigger.FactorMiningIntradayTriggerStateResponseV1.__name__ not in factor_defs.__all__
    assert trigger.FactorMiningIntradayTriggerStateResponseV1.requires_stock_panel
    assert trigger.FactorMiningIntradayTriggerStateResponseV1.requires_bond_stock_map


def test_requirements_use_exact_tminus1_state_and_price_calendar_anchor() -> None:
    requirements = trigger.FactorMiningIntradayTriggerStateResponseV1.daily_requirements()
    assert [item.source for item in requirements] == ["market_cbond.daily_price", "market_cbond.daily_base"]
    assert requirements[0].columns == ("exchange_code", "close_price")
    assert {"trigger_price_revise", "cb_call_price", "cb_put_price", "in_trigger_process"}.issubset(requirements[1].columns)
    assert {"trigger_cum_days_revise", "trigger_reach_days_revise", "stock_code"}.issubset(requirements[1].columns)


def test_all_dynamic_signals_build_without_inf_and_nearcompletion_is_gated() -> None:
    frame = _build()
    ordinary = [*trigger._TRIGGER_TRAJECTORY_SIGNALS, *trigger._APPROACH_RESPONSE_SIGNALS, *trigger._CORRIDOR_SIGNALS]

    assert frame.columns.tolist() == [entry.signal for entry in trigger.intraday_trigger_state_response_catalog()]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame[ordinary].notna().sum(axis=0) >= len(BONDS) - 1).all()
    assert (frame[list(trigger._NEAR_COMPLETION_SIGNALS)].notna().sum(axis=0) >= len(BONDS) // 2 - 1).all()
    assert (frame.nunique(dropna=True) > 1).all()
    assert frame.loc[(SCORE, BONDS[0]), list(trigger._NEAR_COMPLETION_SIGNALS)].notna().all()
    assert frame.loc[(SCORE, BONDS[1]), list(trigger._NEAR_COMPLETION_SIGNALS)].isna().all()


def test_post_cutoff_future_daily_and_future_map_are_ignored() -> None:
    baseline = _build()
    contaminated = _build(
        bond_panel=_with_post_cutoff(_panel(BONDS, stock=False), 50.0),
        stock_panel=_with_post_cutoff(_panel(STOCKS, stock=True), 0.02),
        mapping=pd.concat([_mapping(), _mapping(future=True)], ignore_index=True),
    )
    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_stale_tminus1_state_and_mapping_mismatch_fail_closed_per_code() -> None:
    sources = _daily_sources()
    prior = SCORE.normalize() - pd.offsets.BDay(1)
    stale = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == _bare(BONDS[0]))
            & (sources["market_cbond.daily_base"]["trade_date"] == prior)
        )
    ].copy()
    frame = _build(daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": stale})
    assert frame.loc[(SCORE, BONDS[0])].isna().all()
    assert frame.loc[(SCORE, BONDS[1])].notna().any()

    mismatch = _build(mapping=_mapping(mismatch=BONDS[0]))
    assert mismatch.loc[(SCORE, BONDS[0])].isna().all()


def test_missing_field_duplicate_prior_row_and_future_only_map_are_explicit() -> None:
    kernel = trigger.FactorMiningIntradayTriggerStateResponseV1()
    sources = _daily_sources()
    missing = sources["market_cbond.daily_base"].drop(columns=["trigger_price_revise"])
    with pytest.raises(KeyError, match="trigger_price_revise"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(BONDS, stock=False),
                stock_panel=_panel(STOCKS, stock=True),
                bond_stock_map=_mapping(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": missing},
                params={"signal": "itsr_trigger_distance_segment_convergence"},
            )
        )
    duplicate = pd.concat([sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(BONDS, stock=False),
                stock_panel=_panel(STOCKS, stock=True),
                bond_stock_map=_mapping(),
                daily_data={"market_cbond.daily_price": duplicate, "market_cbond.daily_base": sources["market_cbond.daily_base"]},
                params={"signal": "itsr_trigger_distance_segment_convergence"},
            )
        )
    assert _build(mapping=_mapping(future=True)).isna().all().all()


def test_gaps_do_not_create_response_edges_and_path_does_not_cross_lunch() -> None:
    stock = _panel(STOCKS, stock=True).reset_index()
    gapped = stock.loc[stock["trade_time"].dt.minute % 10 == 0].set_index(["dt", "code", "seq"])
    frame = _build(stock_panel=gapped)
    assert frame[list(trigger._APPROACH_RESPONSE_SIGNALS)].isna().all().all()

    day = SCORE.normalize()
    times = [day + pd.Timedelta(hours=9, minutes=30 + 5 * index) for index in range(6)] + [day + pd.Timedelta(hours=13, minutes=5 * index) for index in range(6)]
    stock_values = np.asarray([11.5, 11.7, 11.9, 12.1, 12.0, 11.8, 11.9, 12.0, 12.2, 12.1, 12.0, 11.9])
    bond_values = np.asarray([115.0, 116.0, 117.0, 118.0, 117.0, 116.0, 117.0, 118.0, 119.0, 118.0, 117.0, 116.0])
    bond = pd.DataFrame({"dt": SCORE, "code": BONDS[0], "seq": range(12), "trade_time": times, "last": bond_values}).set_index(["dt", "code", "seq"])
    stock_path = pd.DataFrame({"dt": SCORE, "code": STOCKS[0], "seq": range(12), "trade_time": times, "last": stock_values}).set_index(["dt", "code", "seq"])
    state = trigger.PriorTriggerState(STOCKS[0], 12.0, 130.0, 100.0, True, 2.0)
    path = trigger._trigger_path(
        trigger._strict_physical_frame(bond, score_date=day, owner="panel"),
        trigger._strict_physical_frame(stock_path, score_date=day, owner="stock_panel"),
        state=state,
    )
    assert path["trigger_distance"].to_numpy() == pytest.approx(np.log(stock_values / 12.0))
    assert path.loc[day + pd.Timedelta(hours=13), "__contiguous"] == np.False_
    assert (path.index.to_series().diff().dropna() == pd.Timedelta(hours=3, minutes=5)).any()
