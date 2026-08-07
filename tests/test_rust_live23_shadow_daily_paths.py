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
from cbond_on.domain.factors.defs.research_factor_mining_daily_contract_stock_v1 import (
    FactorMiningDailyContractStockV1,
)
from cbond_on.domain.factors.defs.research_factor_mining_daily_ohlc_wick_path_asymmetry_v1 import (
    FactorMiningDailyOhlcWickPathAsymmetryV1,
)
from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.infra.factors.rust_legacy_reference import (
    build_factor_frame_rust_legacy_reference,
)


_SHADOW_SITE_ENV = "CBOND_ON_RUST23_SHADOW_SITE"
_SCORE_DAY = pd.Timestamp("2026-08-06")
_CODES = ("110001.SH", "123001.SZ")

_PATH_SPECS = (
    FactorSpec(
        name="dret_drawup_drawdown_asym",
        factor="factor_mining_daily_catalog_v1",
        params={"signal": "dret_drawup_drawdown_asym"},
    ),
    FactorSpec(
        name="dohw_mean_wick_asymmetry60",
        factor="factor_mining_daily_ohlc_wick_path_asymmetry_v1",
        params={"signal": "dohw_mean_wick_asymmetry60"},
    ),
    FactorSpec(
        name="dohw_intraday_sign_range_asymmetry60",
        factor="factor_mining_daily_ohlc_wick_path_asymmetry_v1",
        params={"signal": "dohw_intraday_sign_range_asymmetry60"},
    ),
    FactorSpec(
        name="bstk_tail_cocrash_residual20",
        factor="factor_mining_daily_contract_stock_v1",
        params={"signal": "bstk_tail_cocrash_residual20"},
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


def _panel(*, label_only_code: bool = True) -> pd.DataFrame:
    rows: list[tuple[pd.Timestamp, str, int, pd.Timestamp]] = []
    for code in _CODES:
        rows.extend(
            [
                (_SCORE_DAY, code, 0, _SCORE_DAY.replace(hour=14, minute=28)),
                (_SCORE_DAY, code, 1, _SCORE_DAY.replace(hour=14, minute=29)),
            ]
        )
    if label_only_code:
        rows.append(
            (
                _SCORE_DAY,
                "999999.SH",
                0,
                _SCORE_DAY - pd.Timedelta(days=1),
            )
        )
    index = pd.MultiIndex.from_tuples(
        [(dt, code, seq) for dt, code, seq, _ in rows], names=["dt", "code", "seq"]
    )
    panel = pd.DataFrame(
        {
            "trade_time": [time for _, _, _, time in rows],
            "last": [100.0 + number for number in range(len(rows))],
        },
        index=index,
    )
    panel.attrs["__build_day__"] = _SCORE_DAY.date().isoformat()
    return panel


def _daily_data() -> dict[str, pd.DataFrame]:
    """Complete mixed source contract for catalog, strict OHLC, and bstk."""

    dates = pd.bdate_range(end=_SCORE_DAY - pd.Timedelta(days=1), periods=65)
    base_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    twap_rows: list[dict[str, object]] = []
    for code_number, (code, exchange) in enumerate((("110001", "XSHG"), ("123001", "XSHE"))):
        for position, trade_date in enumerate(dates):
            previous = 100.0 + code_number * 7.0
            close_return = ((position * 7 + code_number * 3) % 13 - 6) * 0.0011
            open_return = ((position * 5 + code_number) % 11 - 5) * 0.0008
            opened = previous * float(np.exp(open_return))
            close = previous * float(np.exp(close_return))
            high = max(opened, close) * (1.003 + (position % 3) * 0.0002)
            low = min(opened, close) * (0.996 - (position % 2) * 0.0001)
            amount = 100_000.0 + position * 91.0 + code_number * 37.0
            volume = 1_000.0 + position * 13.0 + code_number * 5.0
            deal = 100.0 + position * 1.7 + code_number

            price_rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "exchange_code": exchange,
                    "prev_close_price": previous,
                    "act_prev_close_price": previous,
                    "close_price": close,
                    "open_price": opened,
                    "high_price": high,
                    "low_price": low,
                    "volume": volume,
                    "amount": amount,
                    "deal": deal,
                }
            )

            stock_previous = 50.0 + code_number * 3.0
            stock_return = ((position * 11 + code_number * 2) % 17 - 8) * 0.0013
            stock_close = stock_previous * float(np.exp(stock_return))
            bond_previous = 100.0 + code_number * 4.0
            bond_return = 0.0005 + 0.71 * stock_return + ((position % 5) - 2) * 0.00037
            bond_close = bond_previous * float(np.exp(bond_return))
            base_rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "exchange_code": exchange,
                    "year_to_mat": 3.0,
                    "remain_size": 10.0 + code_number,
                    "current_yield": 0.02 + position * 0.00001,
                    "cb_conv_price": 10.0,
                    "turnover_rate": 0.05,
                    "stock_code": "600000.SH",
                    "stock_close_price": stock_close,
                    "bond_prem_ratio": 0.08 + position * 0.0001,
                    "debt_puredebt_ratio": 0.30 + position * 0.0001,
                    "puredebt_prem_ratio": 0.10 + position * 0.00005,
                    "conv_value": 100.0,
                    "ytm": 0.02,
                    "duration": 2.0,
                    "modify_duration": 1.8,
                    "convexity": 4.0,
                    "base_rate": 0.01,
                    "stock_volatility": 0.2,
                    "pure_redemption_value": 100.0,
                    "redemption_prem_ratio": 0.01 * (position + 1),
                    "cb_prev_close_price": bond_previous,
                    "cb_close_price": bond_close,
                    "stk_prev_close_price": stock_previous,
                    "stk_close_price": stock_close,
                    "cb_volume": volume,
                    "cb_amount": amount,
                    "cb_deal": deal,
                    "stk_volume": volume * 2.0,
                    "stk_amount": amount * 2.0,
                    "stk_deal": deal * 2.0,
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
            twap_rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "exchange_code": exchange,
                    "twap_0930_0935": previous,
                    "twap_0935_1000": opened,
                    "twap_1000_1030": close,
                    "twap_1100_1130": close,
                    "twap_1300_1330": close,
                    "twap_1330_1400": close,
                    "twap_1400_1430": close,
                    "twap_1430_1442": close,
                    "twap_1430_1500": close,
                    "twap_1442_1457": close,
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
    specs: tuple[FactorSpec, ...],
) -> pd.DataFrame:
    factor_by_kernel = {
        "factor_mining_daily_catalog_v1": FactorMiningDailyCatalogV1,
        "factor_mining_daily_ohlc_wick_path_asymmetry_v1": (
            FactorMiningDailyOhlcWickPathAsymmetryV1
        ),
        "factor_mining_daily_contract_stock_v1": FactorMiningDailyContractStockV1,
    }
    columns: list[pd.Series] = []
    for spec in specs:
        factor = factor_by_kernel[spec.factor](output_col=spec.name)
        value = factor.compute(
            FactorComputeContext(
                panel=panel.copy(),
                daily_data={source: frame.copy() for source, frame in daily_data.items()},
                params=dict(spec.params),
            )
        )
        columns.append(value.rename(spec.name))
    return pd.concat(columns, axis=1).sort_index()


def _assert_exact_shadow_frame(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
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


def test_daily_paths_shadow_matches_python_golden_exactly() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    expected = _python_golden(panel, daily_data, _PATH_SPECS)
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        _PATH_SPECS,
        rust_module=module,
        daily_data=daily_data,
    )

    _assert_exact_shadow_frame(actual, expected)
    assert np.isfinite(expected.loc[(_SCORE_DAY, list(_CODES)), :].to_numpy(dtype="float64")).all()
    assert expected.loc[(_SCORE_DAY, "999999.SH"), _PATH_SPECS[0].name] != expected.loc[
        (_SCORE_DAY, "999999.SH"), _PATH_SPECS[0].name
    ]


def test_catalog_path_keeps_physical_output_index_when_requested_alone() -> None:
    module = _shadow_module()
    panel = _panel(label_only_code=True)
    daily_data = _daily_data()
    specs = (_PATH_SPECS[0],)
    expected = _python_golden(panel, daily_data, specs)
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        specs,
        rust_module=module,
        daily_data=daily_data,
    )

    _assert_exact_shadow_frame(actual, expected)
    assert (_SCORE_DAY, "999999.SH") not in expected.index


def test_paths_shadow_keeps_catalog_optional_and_strict_contexts_separate() -> None:
    module = _shadow_module()
    panel = _panel(label_only_code=False)
    daily_data = _daily_data()
    for source in daily_data.values():
        target = source["code"].eq("110001")
        source.loc[target, "code"] = "110001.XSHG"
        source.loc[target, "exchange_code"] = "UNKNOWN"
    expected = _python_golden(panel, daily_data, _PATH_SPECS)
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        _PATH_SPECS,
        rust_module=module,
        daily_data=daily_data,
    )

    _assert_exact_shadow_frame(actual, expected)
    assert pd.isna(expected.loc[(_SCORE_DAY, "110001.SH"), "dret_drawup_drawdown_asym"])
    assert np.isfinite(
        expected.loc[
            (_SCORE_DAY, "110001.SH"),
            ["dohw_mean_wick_asymmetry60", "dohw_intraday_sign_range_asymmetry60"],
        ].to_numpy(dtype="float64")
    ).all()
