from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs.research_factor_mining_daily_expansion_v1 import (
    FactorMiningDailyExpansionV1,
)
from cbond_on.domain.factors.defs.research_factor_mining_daily_liquidity_channel_composition_v1 import (
    FactorMiningDailyLiquidityChannelCompositionV1,
)
from cbond_on.domain.factors.defs.research_factor_mining_daily_return_liquidity_topology_v1 import (
    FactorMiningDailyReturnLiquidityTopologyV1,
)
from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.infra.factors.rust_legacy_reference import (
    build_factor_frame_rust_legacy_reference,
)


_SHADOW_SITE_ENV = "CBOND_ON_RUST23_SHADOW_SITE"
_SCORE_DAY = pd.Timestamp("2026-08-06")
_CODES = ("110001.SH", "123001.SZ")

_INFO_SPECS = (
    FactorSpec(
        name="lcc_amount_trade_size_information60",
        factor="factor_mining_daily_liquidity_channel_composition_v1",
        params={"signal": "lcc_amount_trade_size_information60"},
    ),
    FactorSpec(
        name="lcc_volume_deal_information60",
        factor="factor_mining_daily_liquidity_channel_composition_v1",
        params={"signal": "lcc_volume_deal_information60"},
    ),
    FactorSpec(
        name="rlmi_return_deal_sign_mutual_information60",
        factor="factor_mining_daily_return_liquidity_topology_v1",
        params={"signal": "rlmi_return_deal_sign_mutual_information60"},
    ),
    FactorSpec(
        name="rjst_amount_joint_transition_entropy60",
        factor="factor_mining_daily_return_liquidity_topology_v1",
        params={"signal": "rjst_amount_joint_transition_entropy60"},
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


def _information_daily_data() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=_SCORE_DAY - pd.Timedelta(days=1), periods=65)
    rows: list[dict[str, object]] = []
    for code_number, (code, exchange) in enumerate((("110001", "XSHG"), ("123001", "XSHE"))):
        for position, trade_date in enumerate(dates):
            previous = 100.0 + code_number
            log_return = float(((position * 5 + code_number * 3) % 7 - 3) * 0.001)
            close = previous * float(np.exp(log_return))
            log_amount = 11.0 + float(((position * 3 + code_number) % 9 - 4) * 0.07)
            log_volume = 8.0 + float(((position * 5 + code_number * 2) % 11 - 5) * 0.06)
            log_deal = 4.0 + float(((position * 7 + code_number) % 13 - 6) * 0.05)
            rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "exchange_code": exchange,
                    "prev_close_price": previous,
                    "close_price": close,
                    "open_price": previous * 1.0002,
                    "high_price": max(previous, close) * 1.001,
                    "low_price": min(previous, close) * 0.999,
                    "volume": float(np.exp(log_volume)),
                    "amount": float(np.exp(log_amount)),
                    "deal": float(np.exp(log_deal)),
                }
            )
    return {"market_cbond.daily_price": pd.DataFrame(rows)}


def _python_golden(
    panel: pd.DataFrame,
    daily_data: dict[str, pd.DataFrame],
    specs: tuple[FactorSpec, ...],
) -> pd.DataFrame:
    factor_by_kernel = {
        "factor_mining_daily_expansion_v1": FactorMiningDailyExpansionV1,
        "factor_mining_daily_liquidity_channel_composition_v1": (
            FactorMiningDailyLiquidityChannelCompositionV1
        ),
        "factor_mining_daily_return_liquidity_topology_v1": (
            FactorMiningDailyReturnLiquidityTopologyV1
        ),
    }
    columns: list[pd.Series] = []
    for spec in specs:
        factor = factor_by_kernel[spec.factor](output_col=spec.name)
        value = factor.compute(
            FactorComputeContext(
                panel=panel.copy(),
                daily_data={key: source.copy() for key, source in daily_data.items()},
                params=dict(spec.params),
            )
        )
        columns.append(value.rename(spec.name))
    return pd.concat(columns, axis=1).sort_index()


def _assert_exact_shadow_frame(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Keep index, column order, NaN mask, and every finite IEEE value exact."""

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


def _replace_panel_code(panel: pd.DataFrame, *, old: str, new: str) -> pd.DataFrame:
    rows = panel.reset_index().copy()
    rows.loc[rows["code"] == old, "code"] = new
    result = rows.set_index(["dt", "code", "seq"]).sort_index()
    result.attrs = dict(getattr(panel, "attrs", {}) or {})
    return result


def test_daily_information_shadow_matches_python_golden_exactly() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _information_daily_data()
    expected = _python_golden(panel, daily_data, _INFO_SPECS)
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        _INFO_SPECS,
        rust_module=module,
        daily_data=daily_data,
    )

    _assert_exact_shadow_frame(actual, expected)
    assert np.isfinite(expected.to_numpy(dtype="float64")).all()


@pytest.mark.parametrize(
    ("source_code", "source_exchange", "panel_code", "expect_available"),
    [
        ("110001.XSHG", "UNKNOWN", "110001.SH", True),
        ("110001.SHSE", "UNKNOWN", "110001.SH", True),
        ("110001", "UNKNOWN", "110001.SH", False),
        (None, "XSHG", "110001.SH", False),
        ("110001", "XSHG", "110001", False),
    ],
)
def test_daily_information_shadow_uses_strict_code_contract(
    source_code: object,
    source_exchange: object,
    panel_code: str,
    expect_available: bool,
) -> None:
    module = _shadow_module()
    panel = _replace_panel_code(_panel(), old="110001.SH", new=panel_code)
    daily_data = _information_daily_data()
    price = daily_data["market_cbond.daily_price"].copy()
    target = price["code"].eq("110001")
    price.loc[target, "code"] = source_code
    price.loc[target, "exchange_code"] = source_exchange
    daily_data = {"market_cbond.daily_price": price}
    spec = _INFO_SPECS[0]

    expected = _python_golden(panel, daily_data, (spec,))
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        [spec],
        rust_module=module,
        daily_data=daily_data,
    )

    _assert_exact_shadow_frame(actual, expected)
    assert pd.notna(expected.loc[(_SCORE_DAY, panel_code), spec.name]) is expect_available


@pytest.mark.parametrize(
    ("spec", "columns"),
    [
        (
            _INFO_SPECS[0],
            ("trade_date", "code", "exchange_code", "volume", "amount", "deal"),
        ),
        (
            _INFO_SPECS[2],
            (
                "trade_date",
                "code",
                "exchange_code",
                "prev_close_price",
                "close_price",
                "amount",
                "deal",
            ),
        ),
    ],
)
def test_daily_information_shadow_keeps_selected_source_contract(
    spec: FactorSpec,
    columns: tuple[str, ...],
) -> None:
    module = _shadow_module()
    panel = _panel()
    full = _information_daily_data()["market_cbond.daily_price"]
    daily_data = {"market_cbond.daily_price": full.loc[:, list(columns)].copy()}
    expected = _python_golden(panel, daily_data, (spec,))
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        [spec],
        rust_module=module,
        daily_data=daily_data,
    )

    _assert_exact_shadow_frame(actual, expected)


def test_daily_information_shadow_keeps_p1_optional_and_information_strict_contexts_separate() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _information_daily_data()
    price = daily_data["market_cbond.daily_price"].copy()
    target = price["code"].eq("110001")
    price.loc[target, "code"] = "110001.XSHG"
    price.loc[target, "exchange_code"] = "UNKNOWN"
    daily_data = {"market_cbond.daily_price": price}
    p1 = FactorSpec(
        name="dret_volatility_20",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dret_volatility_20"},
    )
    specs = (p1, *_INFO_SPECS)

    expected = _python_golden(panel, daily_data, specs)
    actual = build_factor_frame_rust_legacy_reference(
        panel,
        specs,
        rust_module=module,
        daily_data=daily_data,
    )

    _assert_exact_shadow_frame(actual, expected)
    assert pd.isna(expected.loc[(_SCORE_DAY, "110001.SH"), p1.name])
    for spec in _INFO_SPECS:
        assert pd.notna(expected.loc[(_SCORE_DAY, "110001.SH"), spec.name])


def test_daily_information_shadow_fails_for_global_strict_duplicate() -> None:
    module = _shadow_module()
    panel = _panel()
    price = _information_daily_data()["market_cbond.daily_price"]
    duplicate = price.iloc[[0]].copy()
    duplicate.loc[:, "code"] = "999999"
    duplicate.loc[:, "exchange_code"] = "XSHG"
    daily_data = {
        "market_cbond.daily_price": pd.concat([price, duplicate, duplicate], ignore_index=True)
    }

    with pytest.raises(ValueError, match="duplicate strict-prior"):
        build_factor_frame_rust_legacy_reference(
            panel,
            [_INFO_SPECS[0]],
            rust_module=module,
            daily_data=daily_data,
        )
