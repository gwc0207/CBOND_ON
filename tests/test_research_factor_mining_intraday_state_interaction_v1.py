from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_intraday_state_interaction_v1 as interaction
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(18))


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=9, minutes=45),
        pd.Timedelta(hours=10),
        pd.Timedelta(hours=10, minutes=15),
        pd.Timedelta(hours=10, minutes=30),
        pd.Timedelta(hours=10, minutes=45),
        pd.Timedelta(hours=11),
        pd.Timedelta(hours=11, minutes=30),
        pd.Timedelta(hours=13),
        pd.Timedelta(hours=13, minutes=15),
        pd.Timedelta(hours=13, minutes=30),
        pd.Timedelta(hours=13, minutes=45),
        pd.Timedelta(hours=14),
        pd.Timedelta(hours=14, minutes=15),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _asset_panel(code: str, base: float, *, stock: bool) -> pd.DataFrame:
    amount = 0.0
    volume = 0.0
    trades = 0.0
    rows: list[dict[str, object]] = []
    for sequence, clock in enumerate(_times()):
        phase = 0.004 * np.sin(0.71 * sequence + (0.5 if stock else 0.0))
        drift = (0.0007 if stock else 0.0011) * sequence
        price = base * (1.0 + drift + phase)
        amount += 20_000.0 * (1.0 + 0.12 * np.cos(sequence * 0.63 + (0.3 if stock else 0.0)))
        volume += 200.0 * (1.0 + 0.15 * np.sin(sequence * 0.57 + (0.7 if stock else 0.0)))
        trades += 0.0 if sequence == 8 else float(12 + (sequence % 4))
        mid = price * (1.0 + 0.0003 * np.sin(sequence * 1.13 + (0.2 if stock else 0.0)))
        spread = base * (0.00055 + 0.00004 * ((sequence + (1 if stock else 0)) % 4))
        bid_size = float(40 + ((sequence * 5 + (3 if stock else 0)) % 17))
        ask_size = float(36 + ((sequence * 7 + (1 if stock else 0)) % 19))
        rows.append(
            {
                "dt": SCORE,
                "code": code,
                "seq": sequence,
                "trade_time": SCORE.normalize() + clock,
                "pre_close": base * (0.991 if stock else 0.988),
                "open": base * (1.002 if stock else 1.005),
                "last": price,
                "volume": volume,
                "amount": amount,
                "num_trades": trades,
                "ask_price1": mid + spread / 2.0,
                "bid_price1": mid - spread / 2.0,
                "ask_volume1": ask_size,
                "bid_volume1": bid_size,
            }
        )
    return pd.DataFrame(rows)


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    bonds = [_asset_panel(code, 100.0 + 0.8 * index, stock=False) for index, code in enumerate(CODES)]
    stocks = [
        _asset_panel(f"{600000 + index:06d}.SH", 10.0 + 0.2 * index, stock=True)
        for index in range(len(CODES))
    ]
    bond = pd.concat(bonds, ignore_index=True).set_index(["dt", "code", "seq"])
    stock = pd.concat(stocks, ignore_index=True).set_index(["dt", "code", "seq"])
    bond.attrs["__build_day__"] = SCORE.date().isoformat()
    stock.attrs["__build_day__"] = SCORE.date().isoformat()
    return bond, stock


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=70)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    twap_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        cb_previous = 100.0 + 0.8 * code_index
        stock_previous = 10.0 + 0.2 * code_index
        remain_size = 1_000_000_000.0 + 8_000_000.0 * code_index
        for day_index, day in enumerate(days):
            phase = float(day_index + 2 * code_index)
            cb_return = 0.0012 * np.sin(phase / 3.7) + 0.0005 * np.cos(phase / 5.1)
            stock_return = 0.0017 * np.cos(phase / 4.1) + 0.0004 * np.sin(phase / 6.3)
            cb_close = cb_previous * (1.0 + cb_return)
            stock_close = stock_previous * (1.0 + stock_return)
            cb_adjustment = -0.011 * float((day_index + code_index) % 23 == 0)
            stock_adjustment = -0.014 * float((2 * day_index + code_index) % 29 == 0)
            amount = 2_000_000.0 * (1.0 + 0.08 * code_index + 0.16 * np.sin(phase / 5.2))
            if day_index and (day_index + code_index) % 19 == 0:
                remain_size *= 0.93
            if day_index and (2 * day_index + code_index) % 37 == 0:
                remain_size *= 1.012
            active_start = 20 + (code_index % 8)
            active = day_index >= active_start
            revised_progress = float(max(day_index - active_start + 1, 0))
            contract_shift = 0.08 * float((day_index + code_index) % 31 == 0)
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": "SH",
                    "close_price": cb_close,
                    "prev_close_price": cb_previous,
                    "act_prev_close_price": cb_previous * (1.0 + cb_adjustment),
                    "amount": amount,
                }
            )
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": "SH",
                    "pure_redemption_value": 86.0 + 0.3 * code_index + 0.12 * day_index + 0.9 * np.sin(phase / 8.0),
                    "redemption_prem_ratio": 0.06 + 0.003 * code_index + 0.01 * np.sin(phase / 6.0),
                    "convexity": 0.15 + 0.01 * (code_index % 5) + 0.002 * np.sin(phase / 7.0),
                    "duration": 2.0 + 0.08 * (code_index % 7) + 0.002 * day_index,
                    "year_to_mat": 3.1 + 0.09 * (code_index % 6) + 0.003 * day_index,
                    "bond_prem_ratio": 0.18 + 0.006 * code_index + 0.012 * np.sin(phase / 5.3),
                    "ytm": 0.018 + 0.0007 * code_index + 0.002 * np.cos(phase / 4.7),
                    "remain_size": remain_size,
                    "cb_amount": amount,
                    "in_trigger_process": 1.0 if active else -1.0,
                    "trigger_cum_days_revise": revised_progress,
                    "trigger_reach_days_revise": 16.0 + float(code_index % 3),
                    "cb_conv_price": 11.5 + 0.05 * code_index + contract_shift,
                    "cb_put_price": 9.0 + 0.02 * code_index - contract_shift,
                    "cb_call_price": 12.5 + 0.03 * code_index + contract_shift,
                    "stock_close_price": stock_close,
                    "stk_prev_close_price": stock_previous,
                    "stk_act_prev_close_price": stock_previous * (1.0 + stock_adjustment),
                    "turnover_rate": 0.015 + 0.001 * (code_index % 5) + 0.004 * np.cos(phase / 5.0),
                    "stock_code": f"{600000 + code_index:06d}",
                    "stk_amount": 8_000_000.0 * (1.0 + 0.04 * code_index + 0.12 * np.cos(phase / 4.9)),
                    "stk_deal": 2_000.0 + 11.0 * code_index + 7.0 * day_index,
                    "cb_deal": 600.0 + 5.0 * code_index + 4.0 * day_index,
                    "stock_volatility": 0.025 + 0.002 * (code_index % 4) + 0.004 * np.sin(phase / 4.4),
                    "trigger_price_revise": stock_close * (1.04 + 0.01 * np.sin(phase / 6.0)),
                }
            )
            morning = cb_close * (0.997 + 0.001 * np.sin(phase / 3.0))
            late = cb_close * (1.001 + 0.001 * np.cos(phase / 4.0))
            execution = cb_close * (1.002 + 0.001 * np.sin(phase / 5.0))
            twap_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": "SH",
                    "twap_0930_1000": morning,
                    "twap_1400_1430": late,
                    "twap_1442_1457": execution,
                }
            )
            cb_previous = cb_close
            stock_previous = stock_close
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
        "market_cbond.daily_twap": pd.DataFrame(twap_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=interaction.KERNEL_NAME, params={"signal": entry.signal})
        for entry in interaction.factor_mining_catalog()
    ]


def _build(
    *,
    panel: pd.DataFrame | None = None,
    stock_panel: pd.DataFrame | None = None,
    daily_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    default_panel, default_stock = _panels()
    return build_factor_frame(
        default_panel if panel is None else panel,
        _specs(),
        stock_panel=default_stock if stock_panel is None else stock_panel,
        daily_data=_daily_sources() if daily_data is None else daily_data,
    )


def _contaminate_daily(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        score = terminal.copy()
        future = terminal.copy()
        score["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column in {"trade_date", "code", "exchange_code"}:
                continue
            if pd.api.types.is_numeric_dtype(frame[column]):
                score[column] = 1_000_000.0
                future[column] = 2_000_000.0
            elif column == "stock_code":
                score[column] = "000001"
                future[column] = "000002"
        out[source] = pd.concat([frame, score, future], ignore_index=True)
    return out


def _out_of_contract_panel(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    after_cutoff = frame.iloc[[0]].copy()
    after_cutoff["seq"] = 9000
    after_cutoff["trade_time"] = SCORE.normalize() + pd.Timedelta(hours=14, minutes=30)
    after_cutoff["last"] = 100_000.0
    stale = frame.iloc[[1]].copy()
    stale["seq"] = 9001
    stale["trade_time"] = SCORE.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=14, minutes=29)
    stale["last"] = 200_000.0
    out = pd.concat([frame, after_cutoff, stale], ignore_index=True).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = SCORE.date().isoformat()
    return out


def test_catalogue_has_six_distinct_families_and_import_only_registry() -> None:
    entries = interaction.factor_mining_catalog()

    assert len(entries) == 36
    assert len({entry.signal for entry in entries}) == 36
    assert Counter(entry.family for entry in entries) == {
        "isi_defensive_path_state": 6,
        "isi_supply_event_response": 6,
        "isi_call_contract_book_gate": 6,
        "isi_adjustment_discontinuity_repricing": 6,
        "isi_liquidity_execution_memory": 6,
        "isi_cross_asset_daily_regime_gate": 6,
    }
    assert FactorRegistry.get(interaction.KERNEL_NAME) is interaction.FactorMiningIntradayStateInteractionV1
    assert interaction.FactorMiningIntradayStateInteractionV1.__name__ not in factor_operators.__all__
    assert interaction.FactorMiningIntradayStateInteractionV1.requires_stock_panel is True
    assert interaction.FactorMiningIntradayStateInteractionV1.requires_bond_stock_map is False


def test_requirements_are_explicit_and_only_declared_daily_sources_are_used() -> None:
    requirements = interaction.FactorMiningIntradayStateInteractionV1.daily_requirements()
    assert [requirement.source for requirement in requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_base",
        "market_cbond.daily_twap",
    ]
    base = requirements[1]
    assert "stock_code" in base.columns
    assert "trigger_process" not in base.columns
    assert requirements[0].lookback_days >= 65


def test_all_signals_build_without_inf_and_have_cross_sectional_support() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in interaction.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= len(CODES) - 1).all()


def test_score_future_daily_and_nonphysical_panel_rows_do_not_change_outputs() -> None:
    panel, stock_panel = _panels()
    sources = _daily_sources()
    baseline = _build(panel=panel, stock_panel=stock_panel, daily_data=sources)
    contaminated = _build(
        panel=_out_of_contract_panel(panel),
        stock_panel=_out_of_contract_panel(stock_panel),
        daily_data=_contaminate_daily(sources),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_source_duplicate_prior_rows_and_stale_anchor_fail_closed() -> None:
    panel, stock_panel = _panels()
    kernel = interaction.FactorMiningIntradayStateInteractionV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(
            FactorComputeContext(
                panel=panel,
                stock_panel=stock_panel,
                daily_data={},
                params={"signal": "isi_def_tail_eff_x_floor_change5"},
            )
        )

    sources = _daily_sources()
    duplicate = dict(sources)
    duplicate["market_cbond.daily_price"] = pd.concat(
        [sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        _build(panel=panel, stock_panel=stock_panel, daily_data=duplicate)

    stale_code = _bare(CODES[0])
    anchor = sources["market_cbond.daily_base"]["trade_date"].max()
    stale = dict(sources)
    stale["market_cbond.daily_base"] = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == stale_code)
            & (sources["market_cbond.daily_base"]["trade_date"] == anchor)
        )
    ].copy()
    frame = _build(panel=panel, stock_panel=stock_panel, daily_data=stale)
    assert frame.loc[(SCORE, CODES[0])].isna().all()
    assert frame.loc[(SCORE, CODES[1])].notna().any()


def test_representative_formula_and_mapping_are_strict_tminus1() -> None:
    panel, stock_panel = _panels()
    sources = _daily_sources()
    baseline = _build(panel=panel, stock_panel=stock_panel, daily_data=sources)
    modified = _contaminate_daily(sources)
    changed = _build(panel=panel, stock_panel=stock_panel, daily_data=modified)
    pd.testing.assert_frame_equal(changed, baseline, check_exact=True)

    assert np.isfinite(baseline.loc[(SCORE, CODES[0]), "isi_cross_tailcojump_x_residvol"])
    assert np.isfinite(baseline.loc[(SCORE, CODES[1]), "isi_supply_flowaccel_x_shrink1"])
