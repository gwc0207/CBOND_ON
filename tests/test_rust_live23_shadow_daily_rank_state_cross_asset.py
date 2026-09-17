from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.operators.research_factor_mining_daily_asymmetric_state_transitions_v1 import (
    FactorMiningDailyAsymmetricStateTransitionsV1,
)
from cbond_on.domain.factors.operators.research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1 import (
    FactorMiningDailyBondStockCrossSectionalRankConcordanceV1,
)
from cbond_on.domain.factors.operators.research_factor_mining_daily_bond_stock_return_flow_information_v1 import (
    FactorMiningDailyBondStockReturnFlowInformationV1,
)
from cbond_on.domain.factors.operators.research_factor_mining_daily_capacity_rank_coupling_v1 import (
    FactorMiningDailyCapacityRankCouplingV1,
)
from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.infra.factors.rust_legacy_reference import (
    build_factor_frame_rust_legacy_reference,
)


_SHADOW_SITE_ENV = "CBOND_ON_RUST23_SHADOW_SITE"
_SCORE_DAY = pd.Timestamp("2026-08-06")
_BONDS = (
    ("110001", "XSHG", "600001", 0.15, 0),
    ("110002", "SHSE", "600001", 0.75, 0),
    ("127001", "XSHE", "000001", 1.45, 1),
    ("123001", "SZSE", "000002", 2.10, 2),
    ("113001", "XSHG", "600003", 2.75, 3),
)
_CODES = tuple(
    f"{code}.{('SH' if exchange in {'XSHG', 'SHSE'} else 'SZ')}"
    for code, exchange, *_ in _BONDS
)

_SPECS = (
    FactorSpec(
        name="prcn_return_capacity_rank_corr60",
        factor="factor_mining_daily_capacity_rank_coupling_v1",
        params={"signal": "prcn_return_capacity_rank_corr60"},
    ),
    FactorSpec(
        name="ydpt_yield_fall_return_beta60",
        factor="factor_mining_daily_asymmetric_state_transitions_v1",
        params={"signal": "ydpt_yield_fall_return_beta60"},
    ),
    FactorSpec(
        name="bsfst_stock_return_bond_flow_mutual_information60",
        factor="factor_mining_daily_bond_stock_return_flow_information_v1",
        params={"signal": "bsfst_stock_return_bond_flow_mutual_information60"},
    ),
    FactorSpec(
        name="bssrc_bond_stock_rank_correlation60",
        factor="factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1",
        params={"signal": "bssrc_bond_stock_rank_correlation60"},
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


def _panel(codes: tuple[str, ...] = _CODES) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(dt, code, seq) for code in codes for dt, seq in [(_SCORE_DAY, 0), (_SCORE_DAY, 1)]],
        names=["dt", "code", "seq"],
    )
    panel = pd.DataFrame(
        {
            "trade_time": [
                _SCORE_DAY.replace(hour=14, minute=28),
                _SCORE_DAY.replace(hour=14, minute=29),
            ]
            * len(codes),
            "last": np.linspace(100.0, 101.0, len(index)),
        },
        index=index,
    )
    panel.attrs["__build_day__"] = _SCORE_DAY.date().isoformat()
    return panel


def _stock_return(position: int, group: int, rng: np.random.Generator) -> float:
    # Make date-local ranks move through both tails.  The same `group` is used
    # verbatim for the two bonds sharing 600001, which audits the distinct-
    # underlying consistency rule.
    return float(
        0.021 * np.sin(0.37 * position + (0.15, 1.55, 3.0, 4.4)[group])
        + 0.004 * np.cos(0.11 * position + group)
        + rng.normal(0.0, 0.0002)
    )


def _daily_data(seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=_SCORE_DAY - pd.Timedelta(days=1), periods=65)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for position, trade_date in enumerate(dates):
        underlying_returns = {
            group: _stock_return(position, group, rng) for group in range(4)
        }
        for index, (code, exchange, stock_code, phase, group) in enumerate(_BONDS):
            bond_return = float(
                0.018 * np.sin(0.29 * position + phase)
                + 0.006 * np.cos(0.17 * position + 0.3 * index)
                + rng.normal(0.0, 0.00015)
            )
            # Exact tied returns exercise Pandas' average percentile ranks.
            if position % 7 == 0 and index == 1:
                bond_return = next(
                    row["_bond_return"]
                    for row in price_rows[::-1]
                    if row["trade_date"] == trade_date and row["code"] == "110001"
                )
            previous = 100.0 + index
            close = previous * float(np.exp(bond_return))
            amount = float(
                1_000_000.0
                * np.exp(
                    0.49 * np.sin(0.47 * position + 0.61 * index)
                    + 0.13 * np.cos(0.19 * position + index)
                    + rng.normal(0.0, 0.003)
                )
            )
            opened = previous * float(np.exp(bond_return * 0.55 + 0.0003 * ((position % 3) - 1)))
            price_rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "exchange_code": exchange,
                    "prev_close_price": previous,
                    "act_prev_close_price": previous,
                    "open_price": opened,
                    "high_price": max(opened, close) * 1.002,
                    "low_price": min(opened, close) * 0.998,
                    "close_price": close,
                    "amount": amount,
                    "_bond_return": bond_return,
                }
            )
            stock_previous = 50.0 + group
            stock_close = stock_previous * float(np.exp(underlying_returns[group]))
            yield_change = 0.00042 * np.sin(0.51 * position + 0.29 * index)
            current_yield = 0.025 + 0.00013 * position + yield_change
            base_rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "exchange_code": exchange,
                    "remain_size": 10.0 + 0.5 * index + 0.01 * (position % 5),
                    "current_yield": current_yield,
                    "duration": 2.0 + 0.02 * index,
                    "convexity": 4.0 + 0.1 * index,
                    "stock_volatility": 0.2 + 0.003 * (position % 7),
                    "stock_code": stock_code,
                    "stk_prev_close_price": stock_previous,
                    "stk_close_price": stock_close,
                }
            )
    price = pd.DataFrame(price_rows).drop(columns=["_bond_return"])
    return {
        "market_cbond.daily_price": price,
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _python_golden(
    panel: pd.DataFrame,
    daily_data: dict[str, pd.DataFrame],
    specs: tuple[FactorSpec, ...],
) -> pd.DataFrame:
    factor_by_kernel = {
        "factor_mining_daily_capacity_rank_coupling_v1": (
            FactorMiningDailyCapacityRankCouplingV1
        ),
        "factor_mining_daily_asymmetric_state_transitions_v1": (
            FactorMiningDailyAsymmetricStateTransitionsV1
        ),
        "factor_mining_daily_bond_stock_return_flow_information_v1": (
            FactorMiningDailyBondStockReturnFlowInformationV1
        ),
        "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1": (
            FactorMiningDailyBondStockCrossSectionalRankConcordanceV1
        ),
    }
    values: list[pd.Series] = []
    for spec in specs:
        factor = factor_by_kernel[spec.factor](output_col=spec.name)
        value = factor.compute(
            FactorComputeContext(
                panel=panel.copy(),
                daily_data={key: source.copy() for key, source in daily_data.items()},
                params=dict(spec.params),
            )
        )
        values.append(value.rename(spec.name))
    return pd.concat(values, axis=1).sort_index()


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


def _shadow(
    module: object,
    panel: pd.DataFrame,
    daily_data: dict[str, pd.DataFrame],
    specs: tuple[FactorSpec, ...],
) -> pd.DataFrame:
    return build_factor_frame_rust_legacy_reference(
        panel, specs, rust_module=module, daily_data=daily_data
    )


def test_rank_state_cross_asset_shadow_matches_python_golden_exactly() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    expected = _python_golden(panel, daily_data, _SPECS)
    actual = _shadow(module, panel, daily_data, _SPECS)

    _assert_exact_shadow_frame(actual, expected)
    assert np.isfinite(expected.to_numpy(dtype="float64")).all()


@pytest.mark.parametrize("seed", range(1, 21))
def test_rank_state_cross_asset_shadow_random_seeds_are_bit_exact(seed: int) -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data(seed)
    _assert_exact_shadow_frame(
        _shadow(module, panel, daily_data, _SPECS),
        _python_golden(panel, daily_data, _SPECS),
    )


@pytest.mark.parametrize(
    ("spec", "price_columns", "base_columns"),
    [
        (
            _SPECS[0],
            ("trade_date", "code", "exchange_code", "prev_close_price", "close_price", "amount"),
            ("trade_date", "code", "exchange_code", "remain_size"),
        ),
        (
            _SPECS[1],
            (
                "trade_date",
                "code",
                "exchange_code",
                "act_prev_close_price",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
            ),
            (
                "trade_date",
                "code",
                "exchange_code",
                "current_yield",
                "duration",
                "convexity",
                "stock_volatility",
            ),
        ),
        (
            _SPECS[2],
            ("trade_date", "code", "exchange_code", "prev_close_price", "close_price", "amount"),
            ("trade_date", "code", "exchange_code", "stk_prev_close_price", "stk_close_price"),
        ),
        (
            _SPECS[3],
            ("trade_date", "code", "exchange_code", "prev_close_price", "close_price"),
            (
                "trade_date",
                "code",
                "exchange_code",
                "stock_code",
                "stk_prev_close_price",
                "stk_close_price",
            ),
        ),
    ],
)
def test_rank_cross_shadow_keeps_per_spec_source_contract(
    spec: FactorSpec, price_columns: tuple[str, ...], base_columns: tuple[str, ...]
) -> None:
    module = _shadow_module()
    panel = _panel()
    full = _daily_data()
    daily_data = {
        "market_cbond.daily_price": full["market_cbond.daily_price"].loc[:, list(price_columns)].copy(),
        "market_cbond.daily_base": full["market_cbond.daily_base"].loc[:, list(base_columns)].copy(),
    }
    _assert_exact_shadow_frame(
        _shadow(module, panel, daily_data, (spec,)),
        _python_golden(panel, daily_data, (spec,)),
    )


@pytest.mark.parametrize("missing", ("open_price", "high_price", "duration", "stock_volatility"))
def test_ydpt_shadow_requires_full_declared_family_schema(missing: str) -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    source = "market_cbond.daily_price" if missing in daily_data["market_cbond.daily_price"] else "market_cbond.daily_base"
    daily_data[source] = daily_data[source].drop(columns=[missing])
    with pytest.raises(KeyError):
        _python_golden(panel, daily_data, (_SPECS[1],))
    with pytest.raises(KeyError):
        _shadow(module, panel, daily_data, (_SPECS[1],))


def test_rank_cross_shadow_strict_alias_canonicalization_and_output_keys() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    price = daily_data["market_cbond.daily_price"].copy()
    base = daily_data["market_cbond.daily_base"].copy()
    target_price = price["code"].eq("110001")
    target_base = base["code"].eq("110001")
    price.loc[target_price, "code"] = "110001.XSHG"
    price.loc[target_price, "exchange_code"] = "UNKNOWN"
    base.loc[target_base, "code"] = "110001.XSHG"
    base.loc[target_base, "stock_code"] = "600001.XSHG"
    base.loc[target_base, "exchange_code"] = "UNKNOWN"
    daily_data = {"market_cbond.daily_price": price, "market_cbond.daily_base": base}
    expected = _python_golden(panel, daily_data, _SPECS)
    _assert_exact_shadow_frame(_shadow(module, panel, daily_data, _SPECS), expected)

    # The output index retains the raw panel key, but a bare panel code cannot
    # acquire an exchange at lookup time under the strict shared contract.
    bare_panel = _panel(("110001", *_CODES[1:]))
    expected = _python_golden(bare_panel, daily_data, _SPECS)
    actual = _shadow(module, bare_panel, daily_data, _SPECS)
    _assert_exact_shadow_frame(actual, expected)
    assert expected.loc[(_SCORE_DAY, "110001")].isna().all()


def test_rank_cross_shadow_skips_daily_sources_when_output_universe_is_empty() -> None:
    module = _shadow_module()
    panel = _panel()
    panel.attrs["__build_day__"] = (_SCORE_DAY + pd.Timedelta(days=1)).date().isoformat()
    expected = _python_golden(panel, {}, (_SPECS[0],))
    assert expected.empty
    actual = _shadow(module, panel, {}, (_SPECS[0],))
    assert actual.index.levels[0].dtype == expected.index.levels[0].dtype
    assert actual.index.levels[1].dtype == expected.index.levels[1].dtype
    _assert_exact_shadow_frame(actual, expected)


@pytest.mark.parametrize("spec", _SPECS, ids=lambda spec: spec.name)
def test_rank_cross_shadow_strips_requested_signal_before_dispatch(spec: FactorSpec) -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    params = dict(spec.params or {})
    params["signal"] = f"  {params['signal']}  "
    whitespace_spec = FactorSpec(name=spec.name, factor=spec.factor, params=params)
    expected = _python_golden(panel, daily_data, (whitespace_spec,))
    _assert_exact_shadow_frame(
        _shadow(module, panel, daily_data, (whitespace_spec,)), expected
    )


def test_rank_cross_shadow_global_duplicate_and_score_day_rows_follow_python() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    price = daily_data["market_cbond.daily_price"]
    duplicate = price.iloc[[0]].copy()
    duplicate.loc[:, "code"] = "999999.XSHG"
    duplicate.loc[:, "exchange_code"] = "UNKNOWN"
    daily_data["market_cbond.daily_price"] = pd.concat(
        [price, duplicate, duplicate], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate strict-prior"):
        _python_golden(panel, daily_data, (_SPECS[0],))
    with pytest.raises(ValueError, match="duplicate strict-prior"):
        _shadow(module, panel, daily_data, (_SPECS[0],))

    clean = _daily_data()
    score_rows = clean["market_cbond.daily_price"].iloc[[0, 0]].copy()
    score_rows.loc[:, "trade_date"] = _SCORE_DAY
    clean["market_cbond.daily_price"] = pd.concat(
        [clean["market_cbond.daily_price"], score_rows], ignore_index=True
    )
    _assert_exact_shadow_frame(
        _shadow(module, panel, clean, _SPECS), _python_golden(panel, clean, _SPECS)
    )


def test_cross_bsfst_gap_and_bssrc_shared_underlying_error_match_python() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    gap_day = daily_data["market_cbond.daily_price"].iloc[30]["trade_date"]
    price = daily_data["market_cbond.daily_price"]
    daily_data["market_cbond.daily_price"] = price.loc[
        ~((price["code"] == "110001") & (price["trade_date"] == gap_day))
    ].copy()
    _assert_exact_shadow_frame(
        _shadow(module, panel, daily_data, (_SPECS[2], _SPECS[3])),
        _python_golden(panel, daily_data, (_SPECS[2], _SPECS[3])),
    )

    inconsistent = _daily_data()
    base = inconsistent["market_cbond.daily_base"].copy()
    target = base["code"].eq("110002") & base["trade_date"].eq(base["trade_date"].iloc[10])
    base.loc[target, "stk_close_price"] *= 1.05
    inconsistent["market_cbond.daily_base"] = base
    with pytest.raises(ValueError, match="inconsistent strict-prior underlying return"):
        _python_golden(panel, inconsistent, (_SPECS[3],))
    with pytest.raises(ValueError, match="inconsistent strict-prior underlying return"):
        _shadow(module, panel, inconsistent, (_SPECS[3],))


def test_bssrc_checks_global_underlying_consistency_before_invalid_output_lookup() -> None:
    module = _shadow_module()
    # These strict output keys are deliberately invalid.  Python nevertheless
    # computes its global stock-rank source first, so incompatible copies of a
    # shared underlying must still raise instead of quietly yielding NaN.
    panel = _panel(("110001", "123001"))
    daily_data = _daily_data()
    base = daily_data["market_cbond.daily_base"].copy()
    target = base["code"].eq("110002") & base["trade_date"].eq(base["trade_date"].iloc[10])
    base.loc[target, "stk_close_price"] *= 1.05
    daily_data["market_cbond.daily_base"] = base
    with pytest.raises(ValueError, match="inconsistent strict-prior underlying return"):
        _python_golden(panel, daily_data, (_SPECS[3],))
    with pytest.raises(ValueError, match="inconsistent strict-prior underlying return"):
        _shadow(module, panel, daily_data, (_SPECS[3],))


def test_bssrc_inner_join_excludes_unmapped_history_before_tail_selection() -> None:
    """An unmapped historical bond/day must not consume one of the 60 tail rows.

    Python filters empty underlying codes in ``_stock_rank_history`` and then
    inner-joins it to the bond ranks.  Sixteen such rows therefore leave 49
    valid target observations.  Retaining them as NaNs would instead make the
    latest physical 60-row path contain only 44 finite pairs and fail the
    minimum-observation gate.
    """

    module = _shadow_module()
    panel = _panel()
    daily_data = _daily_data()
    base = daily_data["market_cbond.daily_base"].copy()
    dates = pd.Index(base["trade_date"].drop_duplicates()).sort_values()
    unmapped_dates = dates[5:21]
    target = base["code"].eq("110001") & base["trade_date"].isin(unmapped_dates)
    assert int(target.sum()) == 16
    base.loc[target, "stock_code"] = ""
    daily_data["market_cbond.daily_base"] = base

    expected = _python_golden(panel, daily_data, (_SPECS[3],))
    assert np.isfinite(expected.iloc[:, 0].to_numpy(dtype="float64")).all()
    _assert_exact_shadow_frame(_shadow(module, panel, daily_data, (_SPECS[3],)), expected)


def test_rank_cross_shadow_ignores_missing_daily_source_code() -> None:
    """Missing source codes must remain missing through the Python/Rust adapter."""

    module = _shadow_module()
    panel = _panel()
    clean = _daily_data()
    daily_data = _daily_data()
    source_day = daily_data["market_cbond.daily_price"].iloc[12]["trade_date"]

    price = daily_data["market_cbond.daily_price"]
    extra_price = price.loc[
        price["code"].eq("110001") & price["trade_date"].eq(source_day)
    ].copy()
    assert len(extra_price) == 1
    extra_price.loc[:, "code"] = pd.NA
    extra_price.loc[:, "close_price"] = extra_price["prev_close_price"] * np.exp(0.8)
    daily_data["market_cbond.daily_price"] = pd.concat([price, extra_price], ignore_index=True)

    base = daily_data["market_cbond.daily_base"]
    extra_base = base.loc[
        base["code"].eq("110001") & base["trade_date"].eq(source_day)
    ].copy()
    assert len(extra_base) == 1
    extra_base.loc[:, "code"] = pd.NA
    extra_base.loc[:, "stock_code"] = "999999"
    extra_base.loc[:, "stk_close_price"] = extra_base["stk_prev_close_price"] * np.exp(-0.8)
    daily_data["market_cbond.daily_base"] = pd.concat([base, extra_base], ignore_index=True)

    expected = _python_golden(panel, daily_data, (_SPECS[3],))
    baseline = _python_golden(panel, clean, (_SPECS[3],))
    _assert_exact_shadow_frame(expected, baseline)
    _assert_exact_shadow_frame(_shadow(module, panel, daily_data, (_SPECS[3],)), expected)
