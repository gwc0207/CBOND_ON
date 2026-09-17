from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_historical_execution_to_open_transition_state_v1 as transition,
)
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(8)) + tuple(
    f"127{index:03d}.SZ" for index in range(8)
)


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _panel() -> pd.DataFrame:
    rows = [
        {"dt": SCORE, "code": code, "seq": seq} for code in CODES for seq in range(2)
    ]
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=70)
    price_rows: list[dict[str, object]] = []
    twap_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        day_index = np.arange(len(days), dtype="float64")
        phase = day_index + 1.71 * code_index
        pre_execution = 100.0 + 0.55 * code_index + 0.045 * day_index
        execution_return = (
            0.0031 * np.sin(phase / (3.3 + 0.07 * (code_index % 4)))
            + 0.0017 * np.cos((phase + code_index) / 7.1)
            + 0.00035 * ((code_index % 3) - 1)
        )
        execution = pre_execution * np.exp(execution_return)
        morning = np.empty(len(days), dtype="float64")
        morning[0] = execution[0] * np.exp(0.0015 * np.sin(phase[0] / 2.9))
        for index in range(1, len(days)):
            prior_tail = execution_return[index - 1]
            opening_return = (
                0.0022 * np.sin((index + 0.9 * code_index) / 4.4)
                + 0.0015 * np.sign(prior_tail)
                + 0.0008 * np.cos((index + 2.2 * code_index) / 6.8)
            )
            morning[index] = execution[index - 1] * np.exp(opening_return)
        for index, day in enumerate(days):
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "close_price": 99.0 + 0.4 * code_index + 0.09 * index,
                }
            )
            twap_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "twap_1430_1442": pre_execution[index],
                    "twap_1442_1457": execution[index],
                    "twap_0930_1000": morning[index],
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_twap": pd.DataFrame(twap_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=transition.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in transition.factor_mining_catalog()
    ]


def _build(sources: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    return build_factor_frame(
        _panel(), _specs(), daily_data=_daily_sources() if sources is None else sources
    )


def _contaminate(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        score_day = terminal.copy()
        future = terminal.copy()
        score_day["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column not in {"trade_date", "code", "exchange_code"}:
                score_day[column] = 1_000_000.0
                future[column] = 2_000_000.0
        out[source] = pd.concat([frame, score_day, future], ignore_index=True)
    return out


def test_catalogue_has_three_families_and_exact_daily_context_contract() -> None:
    entries = transition.factor_mining_catalog()
    requirements = transition.FactorMiningHistoricalExecutionToOpenTransitionStateV1.daily_requirements()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "historical_execution_open_terminal_transition": 3,
        "historical_execution_open_transition_regime": 3,
        "historical_execution_open_magnitude_coupling": 3,
    }
    assert set(transition.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(transition.KERNEL_NAME)
        is transition.FactorMiningHistoricalExecutionToOpenTransitionStateV1
    )
    assert (
        transition.FactorMiningHistoricalExecutionToOpenTransitionStateV1.__name__
        not in factor_operators.__all__
    )

    assert [item.source for item in requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_twap",
    ]
    assert requirements[0].columns == ("exchange_code",)
    assert requirements[1].columns == (
        "exchange_code",
        "twap_1430_1442",
        "twap_1442_1457",
        "twap_0930_1000",
    )
    assert all(item.lookback_days == transition._LOOKBACK_DAYS for item in requirements)

    context = infer_factor_context_requirements(_specs())
    assert context.daily_required is True
    assert context.stock_panel_required is False
    assert context.bond_stock_map_required is False


def test_all_nine_signals_build_without_inf_and_have_cross_sectional_support() -> None:
    frame = _build()

    assert frame.columns.tolist() == [
        entry.signal for entry in transition.factor_mining_catalog()
    ]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert (frame.notna().sum(axis=0) == len(CODES)).all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.nunique(axis=0) > 1).all()


def test_score_day_and_future_rows_are_ignored_and_daily_price_values_are_not_inputs() -> (
    None
):
    sources = _daily_sources()
    baseline = _build(sources)

    contaminated = _build(_contaminate(sources))
    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)

    price_value_changed = sources["market_cbond.daily_price"].copy()
    price_value_changed["close_price"] = 9_999_999.0
    calendar_only = _build({**sources, "market_cbond.daily_price": price_value_changed})
    pd.testing.assert_frame_equal(calendar_only, baseline, check_exact=True)


def test_latest_eligible_end_pair_is_admitted_but_stale_twap_history_fails_closed() -> (
    None
):
    sources = _daily_sources()
    baseline = _build(sources)
    terminal_day = sources["market_cbond.daily_price"]["trade_date"].max()
    target = CODES[0]

    endpoint_changed = sources["market_cbond.daily_twap"].copy()
    prior_day = sorted(sources["market_cbond.daily_price"]["trade_date"].unique())[-2]
    prior_execution = float(
        endpoint_changed.loc[
            (endpoint_changed["code"] == _bare(target))
            & (endpoint_changed["trade_date"] == prior_day),
            "twap_1442_1457",
        ].iloc[0]
    )
    endpoint_changed.loc[
        (endpoint_changed["code"] == _bare(target))
        & (endpoint_changed["trade_date"] == terminal_day),
        "twap_0930_1000",
    ] = prior_execution * 0.2
    changed = _build({**sources, "market_cbond.daily_twap": endpoint_changed})
    assert not changed.loc[(SCORE, target)].equals(baseline.loc[(SCORE, target)])

    stale = (
        sources["market_cbond.daily_twap"]
        .loc[
            ~(
                (sources["market_cbond.daily_twap"]["code"] == _bare(target))
                & (sources["market_cbond.daily_twap"]["trade_date"] == terminal_day)
            )
        ]
        .copy()
    )
    stale_frame = _build({**sources, "market_cbond.daily_twap": stale})
    assert stale_frame.loc[(SCORE, target)].isna().all()
    assert stale_frame.loc[(SCORE, CODES[1])].notna().all()


def test_missing_columns_duplicate_rows_and_unknown_signal_fail_closed() -> None:
    kernel = transition.FactorMiningHistoricalExecutionToOpenTransitionStateV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={},
                params={"signal": "heot_terminal_joint_surprisal60"},
            )
        )

    sources = _daily_sources()
    missing_twap = sources["market_cbond.daily_twap"].drop(columns=["twap_0930_1000"])
    with pytest.raises(KeyError, match="twap_0930_1000"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={**sources, "market_cbond.daily_twap": missing_twap},
                params={"signal": "heot_terminal_joint_surprisal60"},
            )
        )

    duplicate_twap = pd.concat(
        [
            sources["market_cbond.daily_twap"],
            sources["market_cbond.daily_twap"].iloc[[0]],
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={**sources, "market_cbond.daily_twap": duplicate_twap},
                params={"signal": "heot_terminal_joint_surprisal60"},
            )
        )

    with pytest.raises(KeyError, match="unknown signal"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(), params={"signal": "unknown_transition"}
            )
        )
