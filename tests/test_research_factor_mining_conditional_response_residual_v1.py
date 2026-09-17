from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_conditional_response_residual_v1 as conditional
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(24)) + tuple(
    f"127{index:03d}.SZ" for index in range(24)
)
_CLOCKS = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:00", "13:30", "14:00", "14:29")


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _panel(codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(codes):
        pre_close = 99.0 + 0.73 * code_index
        early = 0.0045 * np.sin(0.37 * code_index) + 0.0004 * ((code_index % 5) - 2)
        late = 0.0050 * np.cos(0.29 * code_index) - 0.0003 * ((code_index % 7) - 3)
        wiggle = 0.0012 * np.sin(0.81 * code_index)
        offsets = (
            0.0,
            0.42 * early + wiggle,
            early,
            0.72 * early - 0.5 * wiggle,
            0.84 * early,
            early + 0.12 * late,
            early + 0.28 * late - 0.4 * wiggle,
            early + 0.68 * late + 0.25 * wiggle,
            early + late,
        )
        increments = np.asarray(
            [
                170.0 + 3.0 * code_index,
                210.0 + 6.0 * (code_index % 7),
                250.0 + 2.0 * code_index,
                180.0 + 5.0 * (code_index % 9),
                225.0 + 4.0 * code_index,
                260.0 + 8.0 * (code_index % 6),
                330.0 + 7.0 * code_index,
                380.0 + 5.0 * (code_index % 11),
            ],
            dtype="float64",
        )
        cumulative_amount = 1_000.0 + np.concatenate(([0.0], np.cumsum(increments)))
        for seq, (clock, offset) in enumerate(zip(_CLOCKS, offsets, strict=False)):
            price = pre_close * float(np.exp(offset))
            spread = (
                0.0009
                + 0.00008 * (code_index % 6)
                + (0.00004 + 0.000015 * np.sin(0.19 * code_index)) * seq
            )
            midpoint = price * (1.0 + 0.00004 * np.cos(0.3 * code_index + seq))
            imbalance = 0.45 * np.sin(0.41 * code_index + 0.9 * seq)
            depth = 460.0 + 13.0 * (code_index % 10) + 9.0 * seq
            rows.append(
                {
                    "dt": SCORE,
                    "code": code,
                    "seq": seq,
                    "trade_time": pd.Timestamp(f"{SCORE.date()} {clock}:00"),
                    "pre_close": pre_close,
                    "last": price,
                    "amount": float(cumulative_amount[seq]),
                    "ask_price1": midpoint * (1.0 + spread / 2.0),
                    "bid_price1": midpoint * (1.0 - spread / 2.0),
                    "ask_volume1": depth * (1.0 - imbalance) / 2.0,
                    "bid_volume1": depth * (1.0 + imbalance) / 2.0,
                }
            )
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources(codes: tuple[str, ...] = CODES) -> dict[str, pd.DataFrame]:
    prior = SCORE.normalize() - pd.offsets.BDay(1)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(codes):
        close = 100.0 + 0.81 * code_index
        previous = close * (1.0 - 0.004 * np.sin(0.23 * code_index) - 0.001)
        price_rows.append(
            {
                "trade_date": prior,
                "code": _bare(code),
                "exchange_code": _exchange(code),
                "close_price": close,
                "prev_close_price": previous,
                "high_price": close * (1.004 + 0.0004 * (code_index % 5)),
                "low_price": close * (0.994 - 0.0003 * (code_index % 4)),
                "amount": 1_000_000.0 + 55_000.0 * code_index + 3_000.0 * (code_index % 3),
                "deal": 1_400.0 + 17.0 * code_index + 5.0 * (code_index % 4),
            }
        )
        base_rows.append(
            {
                "trade_date": prior,
                "code": _bare(code),
                "exchange_code": _exchange(code),
                "bond_prem_ratio": 0.08 + 0.006 * (code_index % 17) + 0.0005 * code_index,
                "duration": 1.4 + 0.11 * (code_index % 12) + 0.01 * code_index,
                "stock_volatility": 0.15 + 0.008 * (code_index % 9) + 0.001 * code_index,
                "turnover_rate": 0.7 + 0.09 * (code_index % 11) + 0.006 * code_index,
                "remain_size": 1_500_000.0 + 120_000.0 * code_index + 7_000.0 * (code_index % 5),
            }
        )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=conditional.KERNEL_NAME, params={"signal": entry.signal})
        for entry in conditional.conditional_response_residual_catalog()
    ]


def _contaminate(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        score = frame.copy()
        future = frame.copy()
        score["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column not in {"trade_date", "code", "exchange_code"}:
                score[column] = 1_000_000.0
                future[column] = 2_000_000.0
        out[source] = pd.concat([frame, score, future], ignore_index=True)
    return out


def _with_after_cutoff_row(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.reset_index().copy()
    added: list[dict[str, object]] = []
    for code in CODES:
        last = out.loc[(out["code"] == code) & (out["seq"] == 8)].iloc[0].to_dict()
        last["seq"] = 99
        last["trade_time"] = pd.Timestamp(f"{SCORE.date()} 14:30:00")
        last["last"] = float(last["last"]) * 5.0
        last["amount"] = float(last["amount"]) * 100.0
        added.append(last)
    result = pd.concat([out, pd.DataFrame(added)], ignore_index=True).set_index(["dt", "code", "seq"])
    result.attrs["__build_day__"] = SCORE.date().isoformat()
    return result


def test_catalogue_has_four_two_signal_families_and_import_only_registration() -> None:
    entries = conditional.conditional_response_residual_catalog()

    assert len(entries) == 8
    assert len({entry.signal for entry in entries}) == 8
    assert Counter(entry.family for entry in entries) == {
        "conditional_price_response_residual": 2,
        "conditional_liquidity_response_residual": 2,
        "conditional_book_response_residual": 2,
        "conditional_state_interaction_residual": 2,
    }
    assert set(conditional.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(conditional.KERNEL_NAME) is conditional.FactorMiningConditionalResponseResidualV1
    assert conditional.FactorMiningConditionalResponseResidualV1.__name__ not in factor_operators.__all__
    assert conditional.FactorMiningConditionalResponseResidualV1.requires_stock_panel is False
    assert conditional.FactorMiningConditionalResponseResidualV1.requires_bond_stock_map is False


def test_requirements_are_context_only_and_require_both_t1_sources() -> None:
    requirements = conditional.FactorMiningConditionalResponseResidualV1.daily_requirements(
        {"signal": "ccr_late_reversal_prior_state_residual"}
    )

    assert [item.source for item in requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_base",
    ]
    assert "high_price" in requirements[0].columns
    assert "bond_prem_ratio" in requirements[1].columns
    assert all(item.lookback_days >= 5 for item in requirements)


def test_all_eight_signals_build_without_inf_or_constants() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())

    assert frame.columns.tolist() == [entry.signal for entry in conditional.conditional_response_residual_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) == len(CODES)).all()
    assert (frame.nunique(dropna=True) > 1).all()


def test_score_day_future_daily_rows_and_after_cutoff_panel_rows_do_not_change_output() -> None:
    sources = _daily_sources()
    baseline = build_factor_frame(_panel(), _specs(), daily_data=sources)
    contaminated = build_factor_frame(
        _with_after_cutoff_row(_panel()),
        _specs(),
        daily_data=_contaminate(sources),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_stale_and_duplicate_daily_inputs_fail_closed() -> None:
    sources = _daily_sources()
    kernel = conditional.FactorMiningConditionalResponseResidualV1()
    params = {"signal": "ccr_late_reversal_prior_state_residual"}
    missing_price = sources["market_cbond.daily_price"].drop(columns=["high_price"])
    with pytest.raises(KeyError, match="high_price"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": missing_price, "market_cbond.daily_base": sources["market_cbond.daily_base"]},
                params=params,
            )
        )

    duplicate = pd.concat(
        [sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": duplicate, "market_cbond.daily_base": sources["market_cbond.daily_base"]},
                params=params,
            )
        )

    stale_base = sources["market_cbond.daily_base"].loc[
        sources["market_cbond.daily_base"]["code"] != _bare(CODES[0])
    ].copy()
    frame = build_factor_frame(
        _panel(),
        _specs(),
        daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": stale_base},
    )
    assert frame.loc[(SCORE, CODES[0])].isna().all()
    assert frame.loc[(SCORE, CODES[1])].notna().all()


def test_book_signal_requires_l1_book_and_unknown_signal_is_explicit() -> None:
    sources = _daily_sources()
    without_book = _panel().drop(columns=["ask_price1", "bid_price1", "ask_volume1", "bid_volume1"])
    with pytest.raises(KeyError, match="ask_price1"):
        conditional.FactorMiningConditionalResponseResidualV1().compute(
            FactorComputeContext(
                panel=without_book,
                daily_data=sources,
                params={"signal": "ccr_spread_shift_prior_state_residual"},
            )
        )
    with pytest.raises(KeyError, match="unknown signal"):
        conditional.FactorMiningConditionalResponseResidualV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "not_a_conditional_signal"})
        )
