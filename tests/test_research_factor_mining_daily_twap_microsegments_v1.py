from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_daily_twap_microsegments_v1 as microsegments
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(20)) + tuple(
    f"127{index:03d}.SZ" for index in range(20)
)


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _panel(codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    panel = pd.DataFrame(
        [{"dt": SCORE, "code": code, "seq": seq} for code in codes for seq in range(2)]
    ).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=66)
    price_rows: list[dict[str, object]] = []
    twap_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        close = 100.0 + 1.4 * code_index
        for day_index, day in enumerate(days):
            phase = float(day_index + 1.9 * code_index)
            close *= 1.0 + 0.0007 * np.sin(phase / 3.2) + 0.0003 * np.cos(phase / 4.7)
            base = close * (0.994 + 0.0002 * np.cos(phase / 3.9))

            lunch_early = base * (1.0 + 0.0010 * np.sin(phase / 2.9))
            lunch_mid = lunch_early * (1.0 + 0.0011 * np.cos(phase / 3.5 + 0.1 * code_index))
            lunch_late = lunch_mid * (1.0 + 0.0013 * np.sin(phase / 2.3 + 0.2 * code_index))
            twap_1100_1130 = (20.0 * lunch_early + 5.0 * lunch_mid + 5.0 * lunch_late) / 30.0
            twap_1120_1130 = (lunch_mid + lunch_late) / 2.0

            reopen_early = lunch_late * (1.0 + 0.0009 * np.cos(phase / 4.1))
            reopen_mid = reopen_early * (1.0 + 0.0012 * np.sin(phase / 2.7 + 0.15 * code_index))
            reopen_late = reopen_mid * (1.0 + 0.0010 * np.cos(phase / 3.1 + 0.25 * code_index))
            twap_1300_1310 = (reopen_early + reopen_mid) / 2.0
            twap_1300_1330 = (5.0 * reopen_early + 5.0 * reopen_mid + 20.0 * reopen_late) / 30.0

            tail_30_33 = reopen_late * (1.0 + 0.0010 * np.sin(phase / 3.3))
            tail_33_36 = tail_30_33 * (1.0 + 0.0012 * np.cos(phase / 2.8 + 0.15 * code_index))
            tail_36_39 = tail_33_36 * (1.0 + 0.0011 * np.sin(phase / 3.6 + 0.25 * code_index))
            tail_39_42 = tail_36_39 * (1.0 + 0.0013 * np.cos(phase / 2.4 + 0.35 * code_index))
            execution = tail_39_42 * (1.0 + 0.0014 * np.sin(phase / 2.6 + 0.4 * code_index))

            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "close_price": close,
                }
            )
            twap_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "twap_1100_1130": twap_1100_1130,
                    "twap_1120_1130": twap_1120_1130,
                    "twap_1125_1130": lunch_late,
                    "twap_1300_1305": reopen_early,
                    "twap_1300_1310": twap_1300_1310,
                    "twap_1300_1330": twap_1300_1330,
                    "twap_1400_1430": tail_30_33 * (1.0 - 0.0008 * np.cos(phase / 4.3)),
                    "twap_1430_1442": (tail_30_33 + tail_33_36 + tail_36_39 + tail_39_42) / 4.0,
                    "twap_1433_1457": (3.0 * tail_33_36 + 3.0 * tail_36_39 + 3.0 * tail_39_42 + 15.0 * execution) / 24.0,
                    "twap_1436_1457": (3.0 * tail_36_39 + 3.0 * tail_39_42 + 15.0 * execution) / 21.0,
                    "twap_1439_1457": (3.0 * tail_39_42 + 15.0 * execution) / 18.0,
                    "twap_1442_1457": execution,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_twap": pd.DataFrame(twap_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=microsegments.KERNEL_NAME, params={"signal": entry.signal})
        for entry in microsegments.daily_twap_microsegments_catalog()
    ]


def _contaminate(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        score = terminal.copy()
        future = terminal.copy()
        score["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column not in {"trade_date", "code", "exchange_code"}:
                score[column] = 1_000_000.0
                future[column] = 2_000_000.0
        out[source] = pd.concat([frame, score, future], ignore_index=True)
    return out


def test_catalogue_has_four_three_signal_families_and_import_only_registration() -> None:
    entries = microsegments.daily_twap_microsegments_catalog()

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "prior_lunch_terminal_microcurve": 3,
        "prior_reopen_micro_acceleration": 3,
        "prior_preexecution_print_compression": 3,
        "prior_tail_execution_persistence": 3,
    }
    assert set(microsegments.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(microsegments.KERNEL_NAME) is microsegments.FactorMiningDailyTwapMicrosegmentsV1
    assert microsegments.FactorMiningDailyTwapMicrosegmentsV1.__name__ not in factor_operators.__all__
    assert microsegments.FactorMiningDailyTwapMicrosegmentsV1.requires_stock_panel is False
    assert microsegments.FactorMiningDailyTwapMicrosegmentsV1.requires_bond_stock_map is False


def test_requirements_are_family_specific_and_strictly_prior_daily() -> None:
    lunch = microsegments.FactorMiningDailyTwapMicrosegmentsV1.daily_requirements(
        {"signal": "dtms_lunch_terminal_curvature"}
    )
    persistence = microsegments.FactorMiningDailyTwapMicrosegmentsV1.daily_requirements(
        {"signal": "dtms_tail_execution_persistence_corr20"}
    )

    assert [item.source for item in lunch] == ["market_cbond.daily_price", "market_cbond.daily_twap"]
    assert "twap_1125_1130" in lunch[1].columns
    assert "twap_1300_1305" not in lunch[1].columns
    assert "twap_1400_1430" in persistence[1].columns
    assert "twap_1439_1457" in persistence[1].columns
    assert all(item.lookback_days >= 66 for item in persistence)


def test_all_signals_build_without_inf_or_constants_on_complete_history() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())

    assert frame.columns.tolist() == [entry.signal for entry in microsegments.daily_twap_microsegments_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= len(CODES) - 1).all()
    assert (frame.nunique(dropna=True) > 1).all()


def test_all_signals_ignore_score_day_and_future_daily_mutations() -> None:
    baseline = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())
    contaminated = build_factor_frame(_panel(), _specs(), daily_data=_contaminate(_daily_sources()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_field_duplicates_and_stale_twap_row_fail_closed() -> None:
    kernel = microsegments.FactorMiningDailyTwapMicrosegmentsV1()
    sources = _daily_sources()
    missing = sources["market_cbond.daily_twap"].drop(columns=["twap_1125_1130"])

    with pytest.raises(KeyError, match="twap_1125_1130"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_twap": missing},
                params={"signal": "dtms_lunch_terminal_curvature"},
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
                daily_data={"market_cbond.daily_price": duplicate, "market_cbond.daily_twap": sources["market_cbond.daily_twap"]},
                params={"signal": "dtms_lunch_terminal_curvature"},
            )
        )

    latest_day = sources["market_cbond.daily_price"]["trade_date"].max()
    stale_twap = sources["market_cbond.daily_twap"].loc[
        ~(
            (sources["market_cbond.daily_twap"]["code"] == _bare(CODES[0]))
            & (sources["market_cbond.daily_twap"]["trade_date"] == latest_day)
        )
    ].copy()
    frame = build_factor_frame(
        _panel(),
        _specs(),
        daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_twap": stale_twap},
    )
    assert frame.loc[(SCORE, CODES[0])].isna().all()
    assert frame.loc[(SCORE, CODES[1])].notna().all()


def test_implied_microsegments_are_exact_and_unknown_signal_is_explicit() -> None:
    lunch_early, lunch_mid, lunch_late = 100.0, 102.0, 105.0
    reopen_early, reopen_mid, reopen_late = 100.0, 102.0, 104.0
    tail_30_33, tail_33_36, tail_36_39, tail_39_42, execution = 100.0, 101.0, 103.0, 104.0, 106.0
    frame = pd.DataFrame(
        {
            "twap_1100_1130": [(20.0 * lunch_early + 5.0 * lunch_mid + 5.0 * lunch_late) / 30.0],
            "twap_1120_1130": [(lunch_mid + lunch_late) / 2.0],
            "twap_1125_1130": [lunch_late],
            "twap_1300_1305": [reopen_early],
            "twap_1300_1310": [(reopen_early + reopen_mid) / 2.0],
            "twap_1300_1330": [(5.0 * reopen_early + 5.0 * reopen_mid + 20.0 * reopen_late) / 30.0],
            "twap_1400_1430": [99.0],
            "twap_1430_1442": [(tail_30_33 + tail_33_36 + tail_36_39 + tail_39_42) / 4.0],
            "twap_1433_1457": [(3.0 * tail_33_36 + 3.0 * tail_36_39 + 3.0 * tail_39_42 + 15.0 * execution) / 24.0],
            "twap_1436_1457": [(3.0 * tail_36_39 + 3.0 * tail_39_42 + 15.0 * execution) / 21.0],
            "twap_1439_1457": [(3.0 * tail_39_42 + 15.0 * execution) / 18.0],
            "twap_1442_1457": [execution],
        }
    )
    values = microsegments._derived_series(frame)

    assert values["lunch_first"][0] == pytest.approx(np.log(lunch_mid / lunch_early))
    assert values["lunch_second"][0] == pytest.approx(np.log(lunch_late / lunch_mid))
    assert values["reopen_first"][0] == pytest.approx(np.log(reopen_mid / reopen_early))
    assert values["reopen_second"][0] == pytest.approx(np.log(reopen_late / reopen_mid))
    assert values["tail_initial"][0] == pytest.approx(np.log(tail_33_36 / tail_30_33))
    assert values["tail_terminal"][0] == pytest.approx(np.log(tail_39_42 / tail_36_39))
    assert values["tail_execution"][0] == pytest.approx(np.log(execution / tail_39_42))
    assert values["tail_bridge"][0] == pytest.approx(np.log(102.0 / 99.0))

    with pytest.raises(KeyError, match="unknown signal"):
        microsegments.FactorMiningDailyTwapMicrosegmentsV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "not_a_twap_microsegment_signal"})
        )
