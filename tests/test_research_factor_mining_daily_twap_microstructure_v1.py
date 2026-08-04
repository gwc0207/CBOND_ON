from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_daily_twap_microstructure_v1 as micro_twap
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
        close = 100.0 + 1.3 * code_index
        for day_index, day in enumerate(days):
            phase = float(day_index + 1.7 * code_index)
            close *= 1.0 + 0.0008 * np.sin(phase / 3.0) + 0.00035 * np.cos(phase / 5.0)
            base = close * (0.995 + 0.0002 * np.sin(phase / 4.0))

            open_5 = base * (1.0 + 0.0010 * np.sin(phase / 2.3))
            open_8 = open_5 * (1.0 + 0.0013 * np.cos(phase / 3.1))
            open_9 = open_8 * (1.0 + 0.0011 * np.sin(phase / 2.7 + 0.2 * code_index))
            twap_0930_0938 = (5.0 * open_5 + 3.0 * open_8) / 8.0
            twap_0930_0939 = (8.0 * twap_0930_0938 + open_9) / 9.0

            close_0 = base * (1.0 + 0.0009 * np.cos(phase / 3.3))
            close_1 = close_0 * (1.0 + 0.0012 * np.sin(phase / 2.9 + 0.1 * code_index))
            close_2 = close_1 * (1.0 + 0.0010 * np.cos(phase / 3.7))
            close_3 = close_2 * (1.0 + 0.0014 * np.sin(phase / 4.2 + 0.3 * code_index))
            twap_1442_1457 = (5.0 * close_1 + 5.0 * close_2 + 5.0 * close_3) / 15.0
            twap_1447_1457 = (5.0 * close_2 + 5.0 * close_3) / 10.0

            morning_start = base * (1.0 + 0.0006 * np.cos(phase / 2.1))
            morning_mid = morning_start * (1.0 + 0.0010 * np.sin(phase / 4.1))
            morning_late = morning_mid * (1.0 + 0.0011 * np.cos(phase / 3.6 + code_index))
            afternoon_start = morning_late * (1.0 + 0.0008 * np.cos(phase / 4.7))
            afternoon_mid = afternoon_start * (1.0 + 0.0012 * np.sin(phase / 3.4))
            afternoon_late = afternoon_mid * (1.0 + 0.0013 * np.cos(phase / 2.8 + 0.2 * code_index))

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
                    "twap_0930_0935": open_5,
                    "twap_0930_0938": twap_0930_0938,
                    "twap_0930_0939": twap_0930_0939,
                    "twap_1430_1442": close_0,
                    "twap_1442_1457": twap_1442_1457,
                    "twap_1447_1457": twap_1447_1457,
                    "twap_1452_1457": close_3,
                    "twap_1000_1030": morning_start,
                    "twap_1030_1100": morning_mid,
                    "twap_1100_1130": morning_late,
                    "twap_1300_1330": afternoon_start,
                    "twap_1330_1400": afternoon_mid,
                    "twap_1400_1430": afternoon_late,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_twap": pd.DataFrame(twap_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=micro_twap.KERNEL_NAME, params={"signal": entry.signal})
        for entry in micro_twap.daily_twap_microstructure_catalog()
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
    entries = micro_twap.daily_twap_microstructure_catalog()

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "prior_opening_microcurve": 3,
        "prior_closing_print_sequence": 3,
        "prior_session_rotation_microstructure": 3,
        "prior_micro_window_state_stability": 3,
    }
    assert set(micro_twap.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(micro_twap.KERNEL_NAME) is micro_twap.FactorMiningDailyTwapMicrostructureV1
    assert micro_twap.FactorMiningDailyTwapMicrostructureV1.__name__ not in factor_defs.__all__
    assert micro_twap.FactorMiningDailyTwapMicrostructureV1.requires_stock_panel is False
    assert micro_twap.FactorMiningDailyTwapMicrostructureV1.requires_bond_stock_map is False


def test_requirements_are_family_specific_and_use_only_prior_daily_inputs() -> None:
    opening = micro_twap.FactorMiningDailyTwapMicrostructureV1.daily_requirements(
        {"signal": "dtwm_open_microcurve"}
    )
    stability = micro_twap.FactorMiningDailyTwapMicrostructureV1.daily_requirements(
        {"signal": "dtwm_open_microcurve_z20"}
    )

    assert [item.source for item in opening] == ["market_cbond.daily_price", "market_cbond.daily_twap"]
    assert "twap_0930_0938" in opening[1].columns
    assert "twap_1430_1442" not in opening[1].columns
    assert "twap_0930_0938" in stability[1].columns
    assert "twap_1452_1457" in stability[1].columns
    assert all(item.lookback_days >= 66 for item in stability)


def test_all_signals_build_without_inf_or_constants_on_complete_history() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())

    assert frame.columns.tolist() == [entry.signal for entry in micro_twap.daily_twap_microstructure_catalog()]
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
    kernel = micro_twap.FactorMiningDailyTwapMicrostructureV1()
    sources = _daily_sources()
    missing = sources["market_cbond.daily_twap"].drop(columns=["twap_0930_0938"])

    with pytest.raises(KeyError, match="twap_0930_0938"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_twap": missing},
                params={"signal": "dtwm_open_microcurve"},
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
                params={"signal": "dtwm_open_microcurve"},
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


def test_implied_micro_windows_and_unknown_signal_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "twap_0930_0935": [100.0],
            "twap_0930_0938": [(5.0 * 100.0 + 3.0 * 102.0) / 8.0],
            "twap_0930_0939": [(8.0 * ((5.0 * 100.0 + 3.0 * 102.0) / 8.0) + 105.0) / 9.0],
            "twap_1430_1442": [100.0],
            "twap_1442_1457": [(5.0 * 101.0 + 5.0 * 103.0 + 5.0 * 104.0) / 15.0],
            "twap_1447_1457": [(5.0 * 103.0 + 5.0 * 104.0) / 10.0],
            "twap_1452_1457": [104.0],
            "twap_1000_1030": [100.0],
            "twap_1030_1100": [101.0],
            "twap_1100_1130": [102.0],
            "twap_1300_1330": [101.0],
            "twap_1330_1400": [102.0],
            "twap_1400_1430": [104.0],
        }
    )
    values = micro_twap._derived_series(frame)

    assert values["open_5_8"][0] == pytest.approx(np.log(102.0 / 100.0))
    assert values["open_8_9"][0] == pytest.approx(np.log(105.0 / 102.0))
    assert values["close_0_1"][0] == pytest.approx(np.log(101.0 / 100.0))
    assert values["close_1_2"][0] == pytest.approx(np.log(103.0 / 101.0))
    assert values["close_2_3"][0] == pytest.approx(np.log(104.0 / 103.0))

    with pytest.raises(KeyError, match="unknown signal"):
        micro_twap.FactorMiningDailyTwapMicrostructureV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "not_a_twap_micro_signal"})
        )
