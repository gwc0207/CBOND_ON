from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_daily_expansion_v1 as daily_expansion
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = ("110001.SH", "110002.SH", "127001.SZ", "127002.SZ")


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _panel() -> pd.DataFrame:
    rows = [
        {"dt": SCORE, "code": code, "seq": seq}
        for code in CODES
        for seq in range(2)
    ]
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=70)
    price_rows: list[dict[str, object]] = []
    twap_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        for index, day in enumerate(days):
            phase = float(index + 3 * code_index)
            close = 100.0 + 7.0 * code_index + 0.16 * index + 0.65 * np.sin(phase / 4.0)
            prev_close = close * (1.0 - 0.0025 * np.sin((phase - 1.0) / 3.0))
            open_px = prev_close * (1.0 + 0.0017 * np.cos(phase / 3.0))
            high = max(open_px, close) * (1.0 + 0.004 + 0.0003 * (index % 3))
            low = min(open_px, close) * (1.0 - 0.003 - 0.0002 * (index % 4))
            volume = 100_000.0 + 9_000.0 * code_index + 650.0 * index + 1_300.0 * np.sin(phase / 5.0)
            amount = volume * close * (1.0 + 0.006 * np.cos(phase / 6.0))
            deal = 900.0 + 50.0 * code_index + 6.0 * index + 9.0 * np.cos(phase / 4.0)
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "prev_close_price": prev_close,
                    "open_price": open_px,
                    "high_price": high,
                    "low_price": low,
                    "close_price": close,
                    "volume": volume,
                    "amount": amount,
                    "deal": deal,
                }
            )

            morning_open = close * (1.0 - 0.005 + 0.0007 * np.sin(phase / 3.0))
            morning_end = morning_open * (1.0 + 0.0013 * np.cos(phase / 4.0))
            noon = morning_end * (1.0 + 0.0009 * np.sin(phase / 5.0))
            afternoon_open = noon * (1.0 + 0.0011 * np.cos(phase / 6.0))
            afternoon_end = afternoon_open * (1.0 + 0.0015 * np.sin(phase / 4.0))
            late = afternoon_end * (1.0 + 0.0012 * np.cos(phase / 5.0))
            execution = late * (1.0 + 0.0008 * np.sin(phase / 3.0))
            twap_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "twap_0930_0935": morning_open,
                    "twap_0935_1000": morning_end,
                    "twap_1100_1130": noon,
                    "twap_1300_1330": afternoon_open,
                    "twap_1400_1430": afternoon_end,
                    "twap_1430_1442": late,
                    "twap_1442_1457": execution,
                }
            )

            duration = 3.8 + 0.13 * code_index - 0.007 * index + 0.025 * np.sin(phase / 7.0)
            maturity = 5.8 + 0.18 * code_index - 0.009 * index + 0.035 * np.cos(phase / 8.0)
            premium = 0.11 + 0.018 * code_index + 0.00035 * index + 0.008 * np.sin(phase / 3.0)
            ytm = 0.012 + 0.0018 * code_index + 0.00008 * index + 0.0013 * np.cos(phase / 5.0)
            current_yield = 0.014 + 0.0016 * code_index + 0.00006 * index + 0.0010 * np.sin(phase / 6.0)
            redemption_premium = 0.055 + 0.011 * code_index + 0.00019 * index + 0.004 * np.cos(phase / 4.0)
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "bond_prem_ratio": premium,
                    "ytm": ytm,
                    "current_yield": current_yield,
                    "base_rate": 0.0075 + 0.0002 * np.sin(phase / 10.0),
                    "redemption_prem_ratio": redemption_premium,
                    "pure_redemption_value": 90.0 + 2.5 * code_index + 0.025 * index + 0.35 * np.sin(phase / 5.0),
                    "duration": duration,
                    "modify_duration": duration * (0.975 + 0.001 * np.cos(phase / 6.0)),
                    "convexity": 0.29 * duration * duration + 0.03 * np.sin(phase / 4.0),
                    "remain_size": 2_000_000_000.0 + 100_000_000.0 * code_index - 1_300_000.0 * index,
                    "turnover_rate": 0.008 + 0.0015 * code_index + 0.00004 * index + 0.0007 * np.cos(phase / 5.0),
                    "year_to_mat": maturity,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_twap": pd.DataFrame(twap_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=daily_expansion.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in daily_expansion.daily_expansion_catalog()
    ]


def _with_score_and_future_outliers(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        score_rows = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        future_rows = score_rows.copy()
        score_rows["trade_date"] = SCORE.normalize()
        future_rows["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        numeric_columns = [
            column
            for column in frame.columns
            if column not in {"trade_date", "code", "exchange_code"}
        ]
        for column_index, column in enumerate(numeric_columns):
            score_rows[column] = 1_000_000.0 + column_index
            future_rows[column] = 2_000_000.0 + column_index
        out[source] = pd.concat([frame, score_rows, future_rows], ignore_index=True)
    return out


def test_daily_expansion_catalogue_is_explicit_import_only_and_has_eight_families() -> None:
    entries = daily_expansion.daily_expansion_catalog()

    assert len(entries) == 64
    assert len({entry.signal for entry in entries}) == 64
    assert Counter(entry.family for entry in entries) == {
        "daily_return_regime_transition": 8,
        "daily_liquidity_quality_stability": 8,
        "daily_twap_session_memory": 8,
        "daily_premium_yield_curve": 8,
        "daily_redemption_hazard_surface": 8,
        "daily_cross_sectional_structural_residual": 8,
        "daily_duration_convexity_structure": 8,
        "daily_maturity_roll_down_state": 8,
    }
    assert {entry.kernel for entry in entries} == {daily_expansion.KERNEL_NAME}
    assert FactorRegistry.get(daily_expansion.KERNEL_NAME) is daily_expansion.FactorMiningDailyExpansionV1
    assert daily_expansion.FactorMiningDailyExpansionV1.__name__ not in factor_defs.__all__
    assert daily_expansion.FactorMiningDailyExpansionV1.requires_stock_panel is False
    assert daily_expansion.FactorMiningDailyExpansionV1.requires_bond_stock_map is False


def test_daily_requirements_are_signal_specific_and_keep_exchange_disambiguation() -> None:
    twap_requirements = daily_expansion.FactorMiningDailyExpansionV1.daily_requirements(
        {"signal": "dtwap_late_ramp20"}
    )

    assert len(twap_requirements) == 1
    requirement = twap_requirements[0]
    assert requirement.source == "market_cbond.daily_twap"
    assert requirement.columns == (
        "exchange_code",
        "twap_0930_0935",
        "twap_0935_1000",
        "twap_1100_1130",
        "twap_1300_1330",
        "twap_1400_1430",
        "twap_1430_1442",
        "twap_1442_1457",
    )
    assert requirement.lookback_days >= 65

    all_requirements = daily_expansion.FactorMiningDailyExpansionV1.daily_requirements()
    assert {item.source for item in all_requirements} == {
        "market_cbond.daily_price",
        "market_cbond.daily_twap",
        "market_cbond.daily_base",
    }
    assert all("exchange_code" in item.columns for item in all_requirements)


def test_all_daily_expansion_signals_build_to_finite_dt_code_series_contract() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())

    assert frame.columns.tolist() == [entry.signal for entry in daily_expansion.daily_expansion_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame.notna().all().all()


def test_all_signals_ignore_score_day_and_future_daily_outliers() -> None:
    baseline = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())
    contaminated = build_factor_frame(
        _panel(),
        _specs(),
        daily_data=_with_score_and_future_outliers(_daily_sources()),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_daily_expansion_fails_fast_for_missing_source_missing_field_and_prior_duplicate() -> None:
    panel = _panel()
    sources = _daily_sources()
    kernel = daily_expansion.FactorMiningDailyExpansionV1()

    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(
            FactorComputeContext(
                panel=panel,
                daily_data={},
                params={"signal": "dret_last_return"},
            )
        )

    no_high = sources["market_cbond.daily_price"].drop(columns=["high_price"])
    with pytest.raises(KeyError, match="high_price"):
        kernel.compute(
            FactorComputeContext(
                panel=panel,
                daily_data={"market_cbond.daily_price": no_high},
                params={"signal": "dret_close_location_last"},
            )
        )

    duplicated = pd.concat(
        [
            sources["market_cbond.daily_price"],
            sources["market_cbond.daily_price"].iloc[[0]],
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=panel,
                daily_data={"market_cbond.daily_price": duplicated},
                params={"signal": "dret_last_return"},
            )
        )


def test_exchange_suffix_prevents_bare_code_collision() -> None:
    sources = _daily_sources()
    collision = sources["market_cbond.daily_price"].copy()
    sh = collision.loc[collision["code"] == "110001"].copy()
    sh["code"] = "127001"
    sh["exchange_code"] = "SH"
    sz = collision.loc[collision["code"] == "127001"].copy()
    sz["code"] = "127001"
    sz["exchange_code"] = "SZ"
    combined = pd.concat([sh, sz], ignore_index=True)
    panel = _panel().loc[pd.IndexSlice[:, ["127001.SZ"], :], :]
    panel.attrs["__build_day__"] = SCORE.date().isoformat()

    out = daily_expansion.FactorMiningDailyExpansionV1().compute(
        FactorComputeContext(
            panel=panel,
            daily_data={"market_cbond.daily_price": combined},
            params={"signal": "dret_last_return"},
        )
    )

    assert out.index.tolist() == [(SCORE, "127001.SZ")]
    assert np.isfinite(out.iloc[0])
