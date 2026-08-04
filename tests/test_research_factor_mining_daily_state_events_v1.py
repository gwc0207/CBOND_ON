from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_daily_state_events_v1 as state_events
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(18)) + tuple(
    f"127{index:03d}.SZ" for index in range(18)
)


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _panel(codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    panel = pd.DataFrame(
        [{"dt": SCORE, "code": code, "seq": sequence} for code in codes for sequence in range(2)]
    ).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=70)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        cb_previous = 100.0 + 0.8 * code_index
        stock_previous = 10.0 + 0.2 * code_index
        remain_size = 1_000_000_000.0 + 7_000_000.0 * code_index
        activation_start = 25 + (code_index % 12)
        reach = 15.0 + float(code_index % 3)
        for day_index, day in enumerate(days):
            phase = float(day_index + 2 * code_index)
            cb_return = 0.0012 * np.sin(phase / 3.1) + 0.0004 * np.cos(phase / 5.7)
            stock_return = 0.0018 * np.cos(phase / 4.3) + 0.0005 * np.sin(phase / 6.1)
            cb_close = cb_previous * (1.0 + cb_return)
            stock_close = stock_previous * (1.0 + stock_return)
            cb_adjustment = -0.012 * float((day_index + code_index) % 23 == 0)
            stock_adjustment = -0.018 * float((2 * day_index + code_index) % 29 == 0)
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "close_price": cb_close,
                    "prev_close_price": cb_previous,
                    "act_prev_close_price": cb_previous * (1.0 + cb_adjustment),
                }
            )

            if day_index > 0 and (day_index + code_index) % 17 == 0:
                remain_size *= 0.92 - 0.01 * (code_index % 3)
            if day_index > 0 and (3 * day_index + code_index) % 41 == 0:
                remain_size *= 1.015
            active = code_index % 4 != 0 and day_index >= activation_start
            progress = float(min(max(day_index - activation_start + 1, 0), int(reach)))
            cb_volume = 60_000.0 * (1.0 + 0.03 * code_index + 0.25 * np.sin(phase / 4.2))
            stock_volume = 320_000.0 * (1.0 + 0.02 * code_index + 0.20 * np.cos(phase / 5.1))
            cb_amount = cb_volume * cb_close * (1.0 + 0.03 * np.sin(phase / 7.0))
            stock_amount = stock_volume * stock_close * (1.0 + 0.04 * np.cos(phase / 8.0))
            cb_deal = 500.0 + 11.0 * code_index + 9.0 * day_index + 7.0 * np.sin(phase / 5.0)
            stock_deal = 1_800.0 + 13.0 * code_index + 8.0 * day_index + 13.0 * np.cos(phase / 6.0)
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "remain_size": remain_size,
                    "cb_amount": cb_amount,
                    "in_trigger_process": 1.0 if active else -1.0,
                    "trigger_cum_days_revise": progress,
                    "trigger_reach_days_revise": reach,
                    "stk_prev_close_price": stock_previous,
                    "stk_act_prev_close_price": stock_previous * (1.0 + stock_adjustment),
                    "cb_volume": cb_volume,
                    "stk_volume": stock_volume,
                    "stk_amount": stock_amount,
                    "cb_deal": cb_deal,
                    "stk_deal": stock_deal,
                }
            )
            cb_previous = cb_close
            stock_previous = stock_close
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=state_events.KERNEL_NAME, params={"signal": entry.signal})
        for entry in state_events.daily_state_events_catalog()
    ]


def _contaminate(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        score_day = terminal.copy()
        future = terminal.copy()
        score_day["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column in {"trade_date", "code", "exchange_code"}:
                continue
            if pd.api.types.is_numeric_dtype(frame[column]):
                score_day[column] = 1_000_000.0
                future[column] = 2_000_000.0
        out[source] = pd.concat([frame, score_day, future], ignore_index=True)
    return out


def test_catalogue_has_four_distinct_state_event_families_and_24_signals() -> None:
    entries = state_events.daily_state_events_catalog()

    assert len(entries) == 24
    assert len({entry.signal for entry in entries}) == 24
    assert Counter(entry.family for entry in entries) == {
        "capital_supply_transition": 6,
        "call_activation_lifecycle": 6,
        "prior_close_adjustment_discontinuity": 6,
        "bond_stock_quantity_regime": 6,
    }
    assert {entry.kernel for entry in entries} == {state_events.KERNEL_NAME}
    assert FactorRegistry.get(state_events.KERNEL_NAME) is state_events.FactorMiningDailyStateEventsV1
    assert state_events.FactorMiningDailyStateEventsV1.__name__ not in factor_defs.__all__
    assert state_events.FactorMiningDailyStateEventsV1.requires_stock_panel is False
    assert state_events.FactorMiningDailyStateEventsV1.requires_bond_stock_map is False


def test_requirements_are_explicit_and_exclude_rejected_sparse_or_duplicate_fields() -> None:
    requirements = state_events.FactorMiningDailyStateEventsV1.daily_requirements(
        {"signal": "adjustment_net_log_shift"}
    )
    assert [item.source for item in requirements] == ["market_cbond.daily_price", "market_cbond.daily_base"]
    assert "act_prev_close_price" in requirements[0].columns
    assert "stk_act_prev_close_price" in requirements[1].columns
    all_requirements = state_events.FactorMiningDailyStateEventsV1.daily_requirements()
    base = next(item for item in all_requirements if item.source == "market_cbond.daily_base")
    assert "trigger_process" not in base.columns
    assert "trigger_process_revise" not in base.columns
    assert "trigger_type" not in base.columns
    assert "trigger_date" not in base.columns
    assert all(item.lookback_days >= 65 for item in all_requirements)


def test_all_signals_build_without_inf_and_have_cross_sectional_support() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())

    assert frame.columns.tolist() == [entry.signal for entry in state_events.daily_state_events_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= 30).all()


def test_all_signals_ignore_score_day_and_future_daily_rows() -> None:
    baseline = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())
    contaminated = build_factor_frame(_panel(), _specs(), daily_data=_contaminate(_daily_sources()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_fields_duplicate_rows_and_stale_base_fail_closed() -> None:
    kernel = state_events.FactorMiningDailyStateEventsV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(FactorComputeContext(panel=_panel(), daily_data={}, params={"signal": "supply_log_change1"}))

    sources = _daily_sources()
    missing = sources["market_cbond.daily_base"].drop(columns=["remain_size"])
    with pytest.raises(KeyError, match="remain_size"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": missing},
                params={"signal": "supply_log_change1"},
            )
        )

    duplicate_price = pd.concat(
        [sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": duplicate_price, "market_cbond.daily_base": sources["market_cbond.daily_base"]},
                params={"signal": "supply_log_change1"},
            )
        )

    stale_code = _bare(CODES[0])
    latest_day = sources["market_cbond.daily_price"]["trade_date"].max()
    stale_base = sources["market_cbond.daily_base"].loc[
        ~((sources["market_cbond.daily_base"]["code"] == stale_code) & (sources["market_cbond.daily_base"]["trade_date"] == latest_day))
    ].copy()
    frame = build_factor_frame(
        _panel(),
        _specs(),
        daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": stale_base},
    )
    assert frame.loc[(SCORE, CODES[0])].isna().all()
    assert frame.loc[(SCORE, CODES[1])].notna().any()


def test_representative_state_event_formulas() -> None:
    supply = state_events._supply_metrics(
        pd.DataFrame({"remain_size": [100.0] * 19 + [90.0, 81.0], "cb_amount": [1_000.0] * 21})
    )
    assert supply["supply_log_change1"] == pytest.approx(np.log(0.9))
    assert supply["supply_shrink_log1"] == pytest.approx(-np.log(0.9))
    assert supply["supply_event_rate20"] > 0.0

    activation = state_events._activation_metrics(
        pd.DataFrame(
            {
                "in_trigger_process": [-1.0] * 58 + [1.0, 1.0, 1.0],
                "trigger_cum_days_revise": [0.0] * 58 + [1.0, 2.0, 3.0],
                "trigger_reach_days_revise": [15.0] * 61,
            }
        )
    )
    assert activation["call_activation_flag"] == pytest.approx(1.0)
    assert activation["call_active_age"] == pytest.approx(3.0)
    assert activation["call_active_revised_progress"] == pytest.approx(3.0 / 15.0)
    assert activation["call_activation_recency60"] == pytest.approx(1.0 / 3.0)

    adjustment = state_events._adjustment_metrics(
        pd.DataFrame(
            {
                "prev_close_price": [100.0],
                "act_prev_close_price": [99.0],
                "stk_prev_close_price": [10.0],
                "stk_act_prev_close_price": [9.8],
            }
        )
    )
    assert adjustment["adjustment_net_log_shift"] == pytest.approx(np.log(0.99) + np.log(0.98))
    assert adjustment["adjustment_abs_imbalance"] == pytest.approx(abs(np.log(0.99)) - abs(np.log(0.98)))
    assert adjustment["adjustment_cross_asset_gap"] == pytest.approx(np.log(0.99) - np.log(0.98))

    sequence = np.arange(21, dtype="float64")
    quantity = state_events._quantity_metrics(
        pd.DataFrame(
            {
                "cb_volume": 100.0 + 4.0 * sequence + np.sin(sequence),
                "stk_volume": 300.0 + 2.0 * sequence + np.cos(sequence),
                "cb_amount": 10_000.0 + 320.0 * sequence + 3.0 * np.sin(sequence),
                "stk_amount": 30_000.0 + 190.0 * sequence + 4.0 * np.cos(sequence),
                "cb_deal": 40.0 + sequence,
                "stk_deal": 100.0 + 0.7 * sequence,
            }
        )
    )
    assert np.isfinite(quantity["quantity_relative_volume_surprise20"])
    assert np.isfinite(quantity["quantity_volume_coupling20"])
