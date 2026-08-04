from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_underlying_cohort_distribution_v1 as cohort_dist
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(
    f"{111001 + index:06d}.{'SH' if index % 2 == 0 else 'SZ'}"
    for index in range(28)
)


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _panel() -> pd.DataFrame:
    rows = [{"dt": SCORE, "code": code, "seq": seq} for code in CODES for seq in range(2)]
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=70)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        for day_index, day in enumerate(days):
            phase = 0.77 * day_index + 1.41 * code_index
            close = 98.0 + 0.42 * code_index + 0.12 * day_index + 1.6 * np.sin(phase / 3.7)
            prev_close = close * (1.0 - 0.0042 * np.sin((phase - 0.3) / 1.9))
            amount = (
                15_000_000.0
                + 510_000.0 * code_index
                + 42_000.0 * day_index
                + 2_900_000.0 * (1.0 + np.cos(phase / 2.4))
            )
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "prev_close_price": prev_close,
                    "close_price": close,
                    "amount": amount,
                }
            )

            stock_close = 7.0 + 0.16 * code_index + 0.026 * day_index + 0.48 * np.sin(phase / 2.3)
            stock_volatility = 0.14 + 0.004 * (code_index % 7) + 0.005 * (1.0 + np.cos(phase / 4.7))
            stock_amount = (
                510_000_000.0
                + 31_000_000.0 * code_index
                + 1_800_000.0 * day_index
                + 82_000_000.0 * (1.0 + np.sin(phase / 2.6))
            )
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "stock_code": f"{600101 + code_index:06d}.SH",
                    "stock_close_price": stock_close,
                    "stock_volatility": stock_volatility,
                    "stk_amount": stock_amount,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=cohort_dist.KERNEL_NAME, params={"signal": entry.signal})
        for entry in cohort_dist.underlying_cohort_distribution_catalog()
    ]


def _contaminate_with_score_and_future_rows(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        score_rows = terminal.copy()
        future_rows = terminal.copy()
        score_rows["trade_date"] = SCORE.normalize()
        future_rows["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column in {"trade_date", "code", "exchange_code", "stock_code"}:
                continue
            score_rows[column] = 1_000_000.0
            future_rows[column] = 2_000_000.0
        out[source] = pd.concat([frame, score_rows, future_rows], ignore_index=True)
    return out


def test_catalogue_has_two_distribution_families_and_no_peer_mean_signal() -> None:
    entries = cohort_dist.underlying_cohort_distribution_catalog()

    assert len(entries) == 6
    assert len({entry.signal for entry in entries}) == 6
    assert Counter(entry.family for entry in entries) == {
        "underlying_state_distribution": 3,
        "bond_stock_tracking_distribution": 3,
    }
    assert all("mean" not in entry.signal for entry in entries)
    assert {entry.kernel for entry in entries} == {cohort_dist.KERNEL_NAME}
    assert FactorRegistry.get(cohort_dist.KERNEL_NAME) is cohort_dist.FactorMiningUnderlyingCohortDistributionV1
    assert cohort_dist.FactorMiningUnderlyingCohortDistributionV1.__name__ not in factor_defs.__all__
    assert cohort_dist.FactorMiningUnderlyingCohortDistributionV1.requires_stock_panel is False
    assert cohort_dist.FactorMiningUnderlyingCohortDistributionV1.requires_bond_stock_map is False


def test_requirements_are_explicit_and_restricted_to_the_two_t1_sources() -> None:
    requirements = cohort_dist.FactorMiningUnderlyingCohortDistributionV1.daily_requirements(
        {"signal": "utd_peer_tracking_error_skew1"}
    )

    assert [item.source for item in requirements] == ["market_cbond.daily_price", "market_cbond.daily_base"]
    assert requirements[0].columns == ("exchange_code", "prev_close_price", "close_price", "amount")
    assert requirements[1].columns == (
        "exchange_code",
        "stock_code",
        "stock_close_price",
        "stock_volatility",
        "stk_amount",
    )
    assert all(item.lookback_days >= 65 for item in requirements)


def test_all_distribution_signals_build_with_cross_sectional_support_and_without_inf() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(), workers=2)

    assert frame.columns.tolist() == [entry.signal for entry in cohort_dist.underlying_cohort_distribution_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= len(CODES) - 2).all()


def test_score_day_and_future_daily_outliers_cannot_change_outputs() -> None:
    baseline = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources(), workers=2)
    contaminated = build_factor_frame(
        _panel(),
        _specs(),
        daily_data=_contaminate_with_score_and_future_rows(_daily_sources()),
        workers=2,
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_source_field_or_duplicate_strict_history_fails_closed() -> None:
    kernel = cohort_dist.FactorMiningUnderlyingCohortDistributionV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(
            FactorComputeContext(panel=_panel(), daily_data={}, params={"signal": "ucd_peer_stock_return_dispersion1"})
        )

    sources = _daily_sources()
    no_mapping = sources["market_cbond.daily_base"].drop(columns=["stock_code"])
    with pytest.raises(KeyError, match="stock_code"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": no_mapping},
                params={"signal": "ucd_peer_stock_return_dispersion1"},
            )
        )

    duplicate_price = pd.concat(
        [sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": duplicate_price, "market_cbond.daily_base": sources["market_cbond.daily_base"]},
                params={"signal": "ucd_peer_stock_return_dispersion1"},
            )
        )


def test_stale_or_missing_underlying_mapping_is_never_carried_forward() -> None:
    sources = _daily_sources()
    target = CODES[0]
    anchor = sources["market_cbond.daily_price"]["trade_date"].max()
    stale = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == _bare(target))
            & (sources["market_cbond.daily_base"]["exchange_code"] == _exchange(target))
            & (sources["market_cbond.daily_base"]["trade_date"] == anchor)
        )
    ].copy()
    stale_frame = build_factor_frame(
        _panel(),
        _specs(),
        daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": stale},
        workers=2,
    )
    assert stale_frame.loc[(SCORE, target)].isna().all()

    missing_mapping = sources["market_cbond.daily_base"].copy()
    missing_mapping.loc[
        (missing_mapping["code"] == _bare(target))
        & (missing_mapping["exchange_code"] == _exchange(target))
        & (missing_mapping["trade_date"] == anchor),
        "stock_code",
    ] = ""
    missing_mapping_frame = build_factor_frame(
        _panel(),
        _specs(),
        daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": missing_mapping},
        workers=2,
    )
    assert missing_mapping_frame.loc[(SCORE, target)].isna().all()
    assert stale_frame.loc[(SCORE, CODES[-1])].notna().any()


def test_distribution_helpers_measure_shape_not_a_peer_average() -> None:
    snapshot = pd.DataFrame(
        {"value": [-2.0, -1.0, 0.0, 4.0]},
        index=pd.Index(["a", "b", "c", "d"], name="code"),
    )
    peers = ("a", "b", "c", "d")
    weights = np.ones(4, dtype="float64")

    assert cohort_dist._weighted_std(snapshot, peers, weights, "value") == pytest.approx(np.std([-2.0, -1.0, 0.0, 4.0]))
    assert cohort_dist._weighted_skew(snapshot, peers, weights, "value") > 0.0
