from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_daily_incremental_v1 as incremental
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")


def _codes() -> tuple[str, ...]:
    sh = tuple(f"{110000 + index:06d}.SH" for index in range(18))
    sz = tuple(f"{127000 + index:06d}.SZ" for index in range(18))
    return sh + sz


CODES = _codes()


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _panel(codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    rows = [
        {"dt": SCORE, "code": code, "seq": sequence}
        for code in codes
        for sequence in range(2)
    ]
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=76)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    twap_rows: list[dict[str, object]] = []
    phase_patterns = (
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0, -1.0, 1.0),
    )
    for code_index, code in enumerate(CODES):
        previous_close = 100.0 + 2.0 * code_index
        for index, day in enumerate(days):
            phase = float(index + 2 * code_index)
            day_return = 0.0018 * np.sin(phase / 3.0) + 0.0009 * np.cos(phase / 5.0)
            open_px = previous_close * (1.0 + 0.0006 * np.cos(phase / 4.0))
            close = previous_close * (1.0 + day_return)
            high = max(open_px, close) * (1.0 + 0.004 + 0.0002 * (index % 3))
            low = min(open_px, close) * (1.0 - 0.003 - 0.0001 * (index % 4))
            amount = 15_000_000.0 * (
                1.0 + 0.03 * code_index + 0.10 * np.sin(phase / 5.0) + 0.02 * np.cos(index / 3.0)
            )
            volume = amount / max(close, 1.0) * (0.97 + 0.03 * np.cos(phase / 6.0))
            deal = 1_200.0 + 35.0 * code_index + 7.0 * index + 17.0 * np.sin(phase / 4.0)
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "prev_close_price": previous_close,
                    "close_price": close,
                    "high_price": high,
                    "low_price": low,
                    "amount": amount,
                    "volume": volume,
                    "deal": deal,
                }
            )

            ytm = 0.012 + 0.00016 * code_index + 0.00035 * np.sin(phase / 4.0) + 0.00001 * index
            duration = 2.4 + 0.075 * code_index + 0.19 * np.sin(0.73 * code_index) - 0.003 * index
            maturity = 4.5 + 0.05 * (code_index % 7) + 0.11 * np.cos(0.47 * code_index) - 0.004 * index
            remain_size = 1_000_000_000.0 * (
                1.0 + 0.07 * code_index + 0.15 * np.cos(0.61 * code_index) - 0.0002 * index
            )
            premium = (
                0.055
                + 1.45 * ytm
                + 0.0018 * duration
                + 0.0012 * maturity
                + 0.00012 * np.log(remain_size)
                + 0.0025 * np.sin(index / 3.0 + 0.51 * code_index)
            )
            if code_index == 0:
                premium += 0.0015 * index
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "bond_prem_ratio": premium,
                    "ytm": ytm,
                    "current_yield": ytm + 0.0018 + 0.00025 * np.cos(phase / 6.0),
                    "duration": duration,
                    "year_to_mat": maturity,
                    "remain_size": remain_size,
                }
            )

            morning_sign, lunch_sign, afternoon_sign, late_sign, execution_sign = phase_patterns[
                (index + code_index) % len(phase_patterns)
            ]
            morning_open = close * (0.994 + 0.0002 * np.sin(phase))
            morning_end = morning_open * (1.0 + morning_sign * 0.0013)
            noon = morning_end * (1.0 + 0.0002 * np.cos(phase / 3.0))
            afternoon_open = noon * (1.0 + lunch_sign * 0.0011)
            afternoon_end = afternoon_open * (1.0 + afternoon_sign * 0.0014)
            late = afternoon_end * (1.0 + late_sign * 0.0012)
            execution = late * (1.0 + execution_sign * 0.0009)
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
            previous_close = close
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
        "market_cbond.daily_twap": pd.DataFrame(twap_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=incremental.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in incremental.daily_incremental_catalog()
    ]


def _contaminate(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        score_rows = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        future_rows = score_rows.copy()
        score_rows["trade_date"] = SCORE.normalize()
        future_rows["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column_index, column in enumerate(frame.columns):
            if column not in {"trade_date", "code", "exchange_code"}:
                score_rows[column] = 1_000_000.0 + column_index
                future_rows[column] = 2_000_000.0 + column_index
        out[source] = pd.concat([frame, score_rows, future_rows], ignore_index=True)
    return out


def test_incremental_catalogue_has_five_distinct_families_and_explicit_registration() -> None:
    entries = incremental.daily_incremental_catalog()

    assert len(entries) == 28
    assert len({entry.signal for entry in entries}) == 28
    assert Counter(entry.family for entry in entries) == {
        "conditional_premium_repricing_residual": 5,
        "premium_yield_regime_break": 5,
        "lagged_liquidity_price_transmission": 6,
        "drawdown_recovery_timing": 6,
        "twap_phase_transition_entropy": 6,
    }
    assert FactorRegistry.get(incremental.KERNEL_NAME) is incremental.FactorMiningDailyIncrementalV1
    assert incremental.FactorMiningDailyIncrementalV1.__name__ not in factor_operators.__all__
    assert incremental.FactorMiningDailyIncrementalV1.requires_stock_panel is False
    assert incremental.FactorMiningDailyIncrementalV1.requires_bond_stock_map is False


def test_daily_requirements_are_signal_specific_and_strictly_daily() -> None:
    ridge = incremental.FactorMiningDailyIncrementalV1.daily_requirements(
        {"signal": "cprr_premium_delta1_residual"}
    )
    transition = incremental.FactorMiningDailyIncrementalV1.daily_requirements(
        {"signal": "tpt_state_entropy60"}
    )

    assert len(ridge) == 1
    assert ridge[0].source == "market_cbond.daily_base"
    assert ridge[0].lookback_days >= 75
    assert "exchange_code" in ridge[0].columns
    assert "bond_prem_ratio" in ridge[0].columns
    assert len(transition) == 1
    assert transition[0].source == "market_cbond.daily_twap"
    assert "twap_1430_1442" in transition[0].columns
    assert "twap_1442_1457" in transition[0].columns


def test_all_incremental_signals_build_without_inf_on_complete_history() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())

    assert frame.columns.tolist() == [entry.signal for entry in incremental.daily_incremental_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    # A trough on the terminal prior session has no observed recovery path and
    # is intentionally NaN rather than a synthetic zero.  Every family still
    # has ample finite cross-sectional support in this complete fixture.
    assert (frame.notna().sum(axis=0) >= 20).all()


def test_all_incremental_signals_ignore_score_and_future_daily_mutations() -> None:
    baseline = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())
    contaminated = build_factor_frame(
        _panel(),
        _specs(),
        daily_data=_contaminate(_daily_sources()),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_source_field_and_strict_prior_duplicate_fail_closed() -> None:
    panel = _panel()
    sources = _daily_sources()
    kernel = incremental.FactorMiningDailyIncrementalV1()

    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(
            FactorComputeContext(
                panel=panel,
                daily_data={},
                params={"signal": "llpt_signed_flow_lag_beta20"},
            )
        )

    no_amount = sources["market_cbond.daily_price"].drop(columns=["amount"])
    with pytest.raises(KeyError, match="amount"):
        kernel.compute(
            FactorComputeContext(
                panel=panel,
                daily_data={"market_cbond.daily_price": no_amount},
                params={"signal": "llpt_signed_flow_lag_beta20"},
            )
        )

    duplicate = pd.concat(
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
                daily_data={"market_cbond.daily_price": duplicate},
                params={"signal": "drt_drawdown_from_high20"},
            )
        )


def test_exchange_suffix_prevents_bare_code_collision() -> None:
    sources = _daily_sources()
    price = sources["market_cbond.daily_price"]
    sh = price.loc[price["code"] == _bare(CODES[0])].copy()
    sz = price.loc[price["code"] == _bare(CODES[18])].copy()
    sh["code"] = "999999"
    sh["exchange_code"] = "SH"
    sz["code"] = "999999"
    sz["exchange_code"] = "SZ"
    collision = pd.concat([sh, sz], ignore_index=True)
    panel = _panel(("999999.SZ",))

    out = incremental.FactorMiningDailyIncrementalV1().compute(
        FactorComputeContext(
            panel=panel,
            daily_data={"market_cbond.daily_price": collision},
            params={"signal": "drt_drawdown_from_high20"},
        )
    )

    assert out.index.tolist() == [(SCORE, "999999.SZ")]
    assert np.isfinite(out.iloc[0])


def test_conditional_ridge_minimum_n_and_rank_guards_return_nan() -> None:
    sources = _daily_sources()
    signal = "cprr_premium_delta1_residual"
    kernel = incremental.FactorMiningDailyIncrementalV1()
    small_codes = CODES[:10]
    small_base = sources["market_cbond.daily_base"].loc[
        sources["market_cbond.daily_base"]["code"].isin([_bare(code) for code in small_codes])
    ].copy()
    small = kernel.compute(
        FactorComputeContext(
            panel=_panel(small_codes),
            daily_data={"market_cbond.daily_base": small_base},
            params={"signal": signal},
        )
    )
    assert small.isna().all()

    rank_bad = sources["market_cbond.daily_base"].copy()
    rank_bad["ytm"] = 0.012
    rank_bad["duration"] = 3.0
    rank_bad["year_to_mat"] = 5.0
    rank_bad["remain_size"] = 1_000_000_000.0
    bad = kernel.compute(
        FactorComputeContext(
            panel=_panel(),
            daily_data={"market_cbond.daily_base": rank_bad},
            params={"signal": signal},
        )
    )
    assert bad.isna().all()


def _regime_frame() -> pd.DataFrame:
    increments = np.array([0.00010, -0.00007, 0.00013, 0.00005, -0.00004] * 16, dtype="float64")
    ytm = 0.010 + np.r_[0.0, np.cumsum(increments)]
    premium = 0.050 + 2.0 * ytm
    return pd.DataFrame(
        {
            "bond_prem_ratio": premium,
            "ytm": ytm,
            "current_yield": ytm + 0.001 + 0.0001 * np.sin(np.arange(len(ytm))),
        }
    )


def _lag_frame() -> pd.DataFrame:
    count = 76
    amount = np.exp(10.0 + 0.13 * np.sin(np.arange(count) / 3.0))
    ret = np.zeros(count, dtype="float64")
    ret[0] = 0.001
    for index in range(1, count):
        ret[index] = 0.0001 * np.sign(ret[index - 1]) * np.log(amount[index - 1])
    previous = 100.0
    rows: list[dict[str, float]] = []
    for index in range(count):
        close = previous * (1.0 + ret[index])
        rows.append(
            {
                "prev_close_price": previous,
                "close_price": close,
                "amount": amount[index],
                "volume": amount[index] / close,
                "deal": 1_000.0 + index,
            }
        )
        previous = close
    return pd.DataFrame(rows)


def _drawdown_frame() -> pd.DataFrame:
    close = np.array(
        [100.0, 99.0, 98.0, 97.0, 96.0, 94.0, 90.0, 91.0, 92.0, 93.0,
         94.0, 93.5, 94.5, 95.0, 95.5, 96.0, 95.0, 95.5, 96.0, 96.0],
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "close_price": close,
            "high_price": close + 1.0,
            "low_price": close - 1.0,
            "amount": np.linspace(1.0, 2.0, len(close)),
        }
    )


def _phase_frame() -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    patterns = ((1.0, 1.0, -1.0, -1.0, 1.0), (1.0, -1.0, 1.0, -1.0, -1.0),
                (-1.0, 1.0, 1.0, 1.0, -1.0), (-1.0, -1.0, -1.0, 1.0, 1.0))
    for index in range(48):
        morning, lunch, afternoon, late, execution = patterns[index % len(patterns)]
        start = 100.0
        morning_end = start * (1.0 + morning * 0.001)
        noon = morning_end
        afternoon_open = noon * (1.0 + lunch * 0.001)
        afternoon_end = afternoon_open * (1.0 + afternoon * 0.001)
        late_px = afternoon_end * (1.0 + late * 0.001)
        execution_px = late_px * (1.0 + execution * 0.001)
        rows.append(
            {
                "twap_0930_0935": start,
                "twap_0935_1000": morning_end,
                "twap_1100_1130": noon,
                "twap_1300_1330": afternoon_open,
                "twap_1400_1430": afternoon_end,
                "twap_1430_1442": late_px,
                "twap_1442_1457": execution_px,
            }
        )
    return pd.DataFrame(rows)


def test_representative_formula_per_family() -> None:
    full = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())
    conditional = full["cprr_premium_delta1_residual"]
    assert conditional.notna().sum() >= incremental._MIN_CROSS_SECTION
    assert conditional.std() > 0.0

    regime = incremental._premium_yield_regime_metrics(_regime_frame())
    assert regime["pyreg_beta_60"] == pytest.approx(2.0, abs=1e-10)

    lagged = incremental._lagged_liquidity_metrics(_lag_frame())
    assert lagged["llpt_signed_flow_lag_beta20"] == pytest.approx(0.0001, abs=1e-10)

    drawdown = incremental._drawdown_recovery_metrics(_drawdown_frame())
    assert drawdown["drt_drawdown_from_high20"] == pytest.approx((96.0 - 101.0) / 101.0)
    assert drawdown["drt_trough_recency20"] == pytest.approx(13.0 / 19.0)

    transition = incremental._twap_phase_transition_metrics(_phase_frame())
    assert transition["tpt_state_entropy60"] > 0.0
    assert np.isfinite(transition["tpt_latest_transition_surprise"])
