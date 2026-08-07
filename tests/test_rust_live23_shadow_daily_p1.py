from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs.research_factor_mining_catalog_v1 import (
    FactorMiningDailyCatalogV1,
)
from cbond_on.domain.factors.defs.research_factor_mining_daily_expansion_v1 import (
    FactorMiningDailyExpansionV1,
)
from cbond_on.domain.factors.defs.research_factor_mining_daily_incremental_v1 import (
    FactorMiningDailyIncrementalV1,
)
from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.infra.factors.rust_legacy_reference import (
    build_factor_frame_rust_legacy_reference,
)


_SHADOW_SITE_ENV = "CBOND_ON_RUST23_SHADOW_SITE"
_SCORE_DAY = pd.Timestamp("2026-08-06")
_CODES = ("110001.SH", "123001.SZ")

_SPECS = (
    FactorSpec(
        name="base_debt_premium_floor_gap",
        factor="factor_mining_daily_catalog_v1",
        params={"signal": "base_debt_premium_floor_gap"},
    ),
    FactorSpec(
        name="dredemption_bondpremium_interaction",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dredemption_bondpremium_interaction"},
    ),
    FactorSpec(
        name="dredemption_premium_z20",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dredemption_premium_z20"},
    ),
    FactorSpec(
        name="dret_volatility_20",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dret_volatility_20"},
    ),
    FactorSpec(
        name="dliq_volume_return_corr20",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dliq_volume_return_corr20"},
    ),
    FactorSpec(
        name="dtwap_morning_slope20",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dtwap_morning_slope20"},
    ),
    FactorSpec(
        name="drt_rebound_from_low20",
        factor="factor_mining_daily_incremental_v1",
        params={"signal": "drt_rebound_from_low20"},
    ),
)


def _shadow_module() -> object:
    raw = os.environ.get(_SHADOW_SITE_ENV, "").strip()
    if not raw:
        pytest.skip(f"set {_SHADOW_SITE_ENV} to run against an isolated Rust wheel")
    site = Path(raw).resolve()
    if not (site / "cbond_on_rust").is_dir():
        raise AssertionError(f"shadow site does not contain cbond_on_rust: {site}")
    for name in tuple(sys.modules):
        if name == "cbond_on_rust" or name.startswith("cbond_on_rust."):
            del sys.modules[name]
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    module = importlib.import_module("cbond_on_rust")
    extension = importlib.import_module("cbond_on_rust.cbond_on_rust")
    assert Path(extension.__file__).resolve().is_relative_to(site)
    assert hasattr(module, "compute_typed_factor_frame")
    return module


def _panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(dt, code, seq) for code in _CODES for dt, seq in [(_SCORE_DAY, 0), (_SCORE_DAY, 1)]],
        names=["dt", "code", "seq"],
    )
    panel = pd.DataFrame(
        {
            "trade_time": [
                _SCORE_DAY.replace(hour=14, minute=28),
                _SCORE_DAY.replace(hour=14, minute=29),
            ]
            * len(_CODES),
            "last": [100.0, 100.1, 101.0, 101.1],
        },
        index=index,
    )
    panel.attrs["__build_day__"] = _SCORE_DAY.date().isoformat()
    return panel


def _daily_data() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=_SCORE_DAY - pd.Timedelta(days=1), periods=24)
    base_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    twap_rows: list[dict[str, object]] = []
    for code_number, (code, exchange) in enumerate((("110001", "XSHG"), ("123001", "XSHE"))):
        for position, trade_date in enumerate(dates):
            level = float(position + 1 + code_number)
            base_rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "exchange_code": exchange,
                    "year_to_mat": 3.0,
                    "remain_size": 10.0,
                    "current_yield": 0.02,
                    "cb_conv_price": 10.0,
                    "turnover_rate": 0.05,
                    "stock_code": "600000.SH",
                    "stock_close_price": 10.0,
                    "bond_prem_ratio": 0.08 + level * 0.001,
                    "debt_puredebt_ratio": 0.30 + level * 0.001,
                    "puredebt_prem_ratio": 0.10 + level * 0.0005,
                    "conv_value": 100.0,
                    "ytm": 0.02,
                    "duration": 2.0,
                    "modify_duration": 1.8,
                    "convexity": 4.0,
                    "base_rate": 0.01,
                    "stock_volatility": 0.2,
                    "pure_redemption_value": 100.0,
                    "redemption_prem_ratio": 0.01 * level,
                    "cb_prev_close_price": 100.0,
                    "cb_close_price": 101.0,
                    "cb_volume": 100.0,
                    "cb_amount": 10_000.0,
                    "cb_deal": 10.0,
                    "stk_volume": 100.0,
                    "stk_amount": 1_000.0,
                    "stk_deal": 10.0,
                    "cb_call_price": 120.0,
                    "trigger_is_price": 1.0,
                    "trigger_cum_days": 2.0,
                    "trigger_reach_days": 15.0,
                    "in_trigger_process": 0.0,
                    "trigger_price_revise": 110.0,
                    "trigger_cum_days_revise": 1.0,
                    "trigger_reach_days_revise": 14.0,
                }
            )
            prev_close = 100.0 + code_number
            close = prev_close * (1.0 + 0.001 * level)
            price_rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "exchange_code": exchange,
                    "prev_close_price": prev_close,
                    "act_prev_close_price": prev_close,
                    "close_price": close,
                    "open_price": prev_close * 1.0005,
                    "high_price": close * 1.001,
                    "low_price": prev_close * 0.999,
                    "volume": float(1000 + position * 17 + code_number),
                    "amount": 100_000.0 + position,
                    "deal": 100.0 + position,
                }
            )
            twap_rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "exchange_code": exchange,
                    "twap_0930_0935": 100.0,
                    "twap_0935_1000": 100.0 + 0.01 * level,
                    "twap_1000_1030": 100.0,
                    "twap_1100_1130": 100.0,
                    "twap_1300_1330": 100.0,
                    "twap_1330_1400": 100.0,
                    "twap_1400_1430": 100.0,
                    "twap_1430_1442": 100.0,
                    "twap_1430_1500": 100.0,
                    "twap_1442_1457": 100.0,
                }
            )
    return {
        "market_cbond.daily_base": pd.DataFrame(base_rows),
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_twap": pd.DataFrame(twap_rows),
    }


def _python_golden(
    panel: pd.DataFrame,
    daily_data: dict[str, pd.DataFrame],
    specs: tuple[FactorSpec, ...] = _SPECS,
) -> pd.DataFrame:
    factor_by_kernel = {
        "factor_mining_daily_catalog_v1": FactorMiningDailyCatalogV1,
        "factor_mining_daily_expansion_v1": FactorMiningDailyExpansionV1,
        "factor_mining_daily_incremental_v1": FactorMiningDailyIncrementalV1,
    }
    columns: list[pd.Series] = []
    for spec in specs:
        factor = factor_by_kernel[spec.factor](output_col=spec.name)
        value = factor.compute(
            FactorComputeContext(
                panel=panel.copy(),
                daily_data={key: value.copy() for key, value in daily_data.items()},
                params=dict(spec.params),
            )
        )
        columns.append(value.rename(spec.name))
    return pd.concat(columns, axis=1).sort_index()


def _assert_exact_shadow_frame(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Require matching labels, NaN masks, and every finite float64 bit."""

    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    assert list(actual.columns) == list(expected.columns)
    for column in expected.columns:
        expected_values = expected[column].to_numpy(dtype="float64")
        actual_values = actual[column].to_numpy(dtype="float64")
        np.testing.assert_array_equal(np.isnan(actual_values), np.isnan(expected_values))
        finite = np.isfinite(expected_values)
        np.testing.assert_array_equal(np.isfinite(actual_values), finite)
        np.testing.assert_array_equal(
            actual_values[finite].view(np.uint64),
            expected_values[finite].view(np.uint64),
        )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True, check_exact=True)


def test_p1_daily_shadow_matches_python_golden_exactly() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    expected = _python_golden(panel, daily_data)
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        _SPECS,
        rust_module=module,
        daily_data=daily_data,
    )
    _assert_exact_shadow_frame(actual, expected)


def test_p1_daily_shadow_fails_for_unrelated_strict_prior_duplicate() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    duplicate = daily_data["market_cbond.daily_price"].iloc[[0]].copy()
    duplicate.loc[:, "code"] = "999999"
    duplicate.loc[:, "exchange_code"] = "XSHG"
    daily_data["market_cbond.daily_price"] = pd.concat(
        [daily_data["market_cbond.daily_price"], duplicate, duplicate],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior"):
        build_factor_frame_rust_legacy_reference(
            panel,
            [_SPECS[3]],
            rust_module=module,
            daily_data=daily_data,
        )


def test_p1_daily_shadow_preserves_terminal_nan_fail_closed() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    latest = daily_data["market_cbond.daily_price"]["trade_date"].max()
    rows = daily_data["market_cbond.daily_price"]
    rows.loc[(rows["trade_date"] == latest) & (rows["code"] == "110001"), "amount"] = np.nan
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        [_SPECS[-1]],
        rust_module=module,
        daily_data=daily_data,
    )
    assert pd.isna(actual.loc[(_SCORE_DAY, "110001.SH"), "drt_rebound_from_low20"])


def test_p1_daily_shadow_keeps_per_factor_source_contracts() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    price_columns = (
        "trade_date",
        "code",
        "exchange_code",
        "prev_close_price",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    )
    minimal_daily = {
        "market_cbond.daily_price": daily_data["market_cbond.daily_price"].loc[:, price_columns]
    }
    spec = _SPECS[3]
    expected = _python_golden(panel, minimal_daily, (spec,))
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        [spec],
        rust_module=module,
        daily_data=minimal_daily,
    )
    _assert_exact_shadow_frame(actual, expected)


def test_p1_daily_shadow_uses_build_day_and_catalog_physical_output_index() -> None:
    module = _shadow_module()
    panel = _panel()
    rolling_index = pd.MultiIndex.from_tuples(
        [(_SCORE_DAY - pd.Timedelta(days=1), "999999.SH", 0)],
        names=["dt", "code", "seq"],
    )
    rolling = pd.DataFrame(
        {"trade_time": [_SCORE_DAY - pd.Timedelta(days=1)], "last": [99.0]},
        index=rolling_index,
    )
    panel = pd.concat([panel, rolling])
    panel.attrs["__build_day__"] = _SCORE_DAY.date().isoformat()
    # The index label is score day but the physical snapshot is an old rolling
    # row.  The catalog kernel excludes it; daily-expansion outputs retain the
    # labelled score-day universe, so the joined base column is missing there.
    panel.index = pd.MultiIndex.from_tuples(
        [
            (dt, code, seq)
            if code != "123001.SZ"
            else (_SCORE_DAY, code, seq)
            for dt, code, seq in panel.index
        ],
        names=["dt", "code", "seq"],
    )
    panel = panel.sort_index()
    panel.loc[(_SCORE_DAY, "123001.SZ", 0), "trade_time"] = _SCORE_DAY - pd.Timedelta(days=1)
    panel.loc[(_SCORE_DAY, "123001.SZ", 1), "trade_time"] = _SCORE_DAY - pd.Timedelta(days=1)
    daily_data = _daily_data()
    specs = (_SPECS[0], _SPECS[3])
    expected = _python_golden(panel, daily_data, specs)
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        specs,
        rust_module=module,
        daily_data=daily_data,
    )
    _assert_exact_shadow_frame(actual, expected)


def _replace_panel_code(panel: pd.DataFrame, *, old: str, new: str) -> pd.DataFrame:
    rows = panel.reset_index().copy()
    rows.loc[rows["code"] == old, "code"] = new
    result = rows.set_index(["dt", "code", "seq"]).sort_index()
    result.attrs = dict(getattr(panel, "attrs", {}) or {})
    return result


@pytest.mark.parametrize(
    ("source_code", "source_exchange", "panel_code", "expect_available"),
    [
        ("110001.XSHG", "UNKNOWN", "110001.XSHG", True),
        ("110001.SHSE", "UNKNOWN", "110001.SHSE", True),
        ("110001.XSHE", "UNKNOWN", "110001.XSHE", True),
        ("110001.SZSE", "UNKNOWN", "110001.SZSE", True),
        ("110001.BSE", "UNKNOWN", "110001.BSE", True),
        ("110001.BJSE", "UNKNOWN", "110001.BJSE", True),
        ("110001", "XSHG", "110001.SH", True),
        ("110001.0", "SHSE", "110001.SH", True),
        ("110001", "UNKNOWN", "110001.SH", False),
        ("110001", None, "110001.SH", False),
        (None, "XSHG", "110001.SH", False),
        ("", "XSHG", "110001.SH", False),
    ],
)
def test_p1_daily_shadow_optional_code_canonicalization_matches_python(
    source_code: object,
    source_exchange: object,
    panel_code: str,
    expect_available: bool,
) -> None:
    """Expansion keeps the same optional-exchange canonical contract as Python."""

    module = _shadow_module()
    panel = _replace_panel_code(_panel(), old="110001.SH", new=panel_code)
    daily_data = _daily_data()
    price = daily_data["market_cbond.daily_price"].copy()
    target = price["code"].eq("110001")
    price.loc[target, "code"] = source_code
    price.loc[target, "exchange_code"] = source_exchange
    minimal_daily = {"market_cbond.daily_price": price}
    spec = _SPECS[3]

    expected = _python_golden(panel, minimal_daily, (spec,))
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        [spec],
        rust_module=module,
        daily_data=minimal_daily,
    )

    _assert_exact_shadow_frame(actual, expected)
    value = expected.loc[(_SCORE_DAY, panel_code), spec.name]
    assert pd.notna(value) is expect_available


def test_p1_catalog_shadow_keeps_optional_exchange_code_contract() -> None:
    """Do not apply strict alias canonicalisation to catalog's optional contract."""

    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    for source in daily_data.values():
        target = source["code"].eq("110001")
        source.loc[target, "code"] = "110001.XSHG"
        source.loc[target, "exchange_code"] = "XSHG"
    spec = _SPECS[0]

    expected = _python_golden(panel, daily_data, (spec,))
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        [spec],
        rust_module=module,
        daily_data=daily_data,
    )

    _assert_exact_shadow_frame(actual, expected)
    assert pd.isna(expected.loc[(_SCORE_DAY, "110001.SH"), spec.name])


def test_p1_shadow_keeps_catalog_and_expansion_optional_contexts_consistent() -> None:
    """A mixed P1 payload keeps the two optional-exchange contracts identical."""

    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    for source in daily_data.values():
        target = source["code"].eq("110001")
        source.loc[target, "code"] = "110001.XSHG"
        source.loc[target, "exchange_code"] = "XSHG"
    specs = (_SPECS[0], _SPECS[3])

    expected = _python_golden(panel, daily_data, specs)
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        specs,
        rust_module=module,
        daily_data=daily_data,
    )

    _assert_exact_shadow_frame(actual, expected)
    assert pd.isna(expected.loc[(_SCORE_DAY, "110001.SH"), _SPECS[0].name])
    assert pd.isna(expected.loc[(_SCORE_DAY, "110001.SH"), _SPECS[3].name])
