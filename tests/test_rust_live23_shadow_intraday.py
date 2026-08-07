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
from cbond_on.domain.factors.defs.research_factor_mining_orderbook_repricing_v1 import (
    FactorMiningOrderbookRepricingV1,
)
from cbond_on.domain.factors.defs.research_factor_mining_quote_execution_dynamics_v1 import (
    FactorMiningQuoteExecutionDynamicsV1,
)
from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.infra.factors.rust_legacy_reference import (
    build_factor_frame_rust_legacy_reference,
)


_SHADOW_SITE_ENV = "CBOND_ON_RUST23_SHADOW_SITE"
_SCORE_DAY = pd.Timestamp("2026-08-06")
_LABEL_TIME = _SCORE_DAY.replace(hour=14, minute=30)
_QED_SPECS = (
    FactorSpec(
        name="qed_prior_quote_location_dispersion",
        factor="factor_mining_quote_execution_dynamics_v1",
        params={"signal": "qed_prior_quote_location_dispersion"},
    ),
    FactorSpec(
        name="qed_prior_quote_tail_penetration",
        factor="factor_mining_quote_execution_dynamics_v1",
        params={"signal": "qed_prior_quote_tail_penetration"},
    ),
    FactorSpec(
        name="qed_prior_quote_lag2_agreement",
        factor="factor_mining_quote_execution_dynamics_v1",
        params={"signal": "qed_prior_quote_lag2_agreement"},
    ),
)
_LRD_SPEC = FactorSpec(
    name="lrd_cross_side_reprice_symmetry",
    factor="factor_mining_orderbook_repricing_v1",
    params={"signal": "lrd_cross_side_reprice_symmetry"},
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


def _clock_timestamp(second: int, *, timezone: str | None = None) -> pd.Timestamp:
    base = pd.Timestamp("2026-08-06 09:30:00", tz=timezone)
    return base + pd.Timedelta(seconds=second)


def _book_fields(index: int, location: float) -> dict[str, float]:
    """Build a valid L1--L5 book whose two ladders reprice in tandem."""

    move = 0.013 * index
    values: dict[str, float] = {}
    for level in range(1, 6):
        values[f"ask_price{level}"] = 101.0 + move + 0.1 * (level - 1)
        values[f"bid_price{level}"] = 99.0 + move - 0.1 * (level - 1)
        values[f"ask_volume{level}"] = 10.0 + ((index + 3 * level) % 7) * 0.25
        values[f"bid_volume{level}"] = 11.0 + ((2 * index + level) % 9) * 0.20
    # Current execution price is relative to the prior L1 midpoint.  The
    # caller overwrites the first row separately because it has no predecessor.
    values["last"] = 100.0 + move + location
    return values


def _panel(
    seed: int = 0,
    *,
    timezone: str | None = None,
    include_nonphysical_label: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = rng.choice(
        np.array([-2.5, -1.0, -0.4, 0.0, 0.6, 1.0, 1.7, 2.2], dtype="float64"),
        size=11,
        replace=True,
    )
    increments = rng.uniform(0.15, 4.0, size=11)
    rows: list[dict[str, object]] = []
    index: list[tuple[pd.Timestamp, str, int]] = []
    count = 0.0
    for seq in range(12):
        location = 0.0 if seq == 0 else float(locations[seq - 1])
        row: dict[str, object] = {
            "trade_time": _clock_timestamp(seq, timezone=timezone),
            "num_trades": count,
        }
        row.update(_book_fields(seq, location))
        if seq == 0:
            row["last"] = 100.0
        else:
            count += float(increments[seq - 1])
            row["num_trades"] = count
            # QED uses the *prior* midpoint, not the current moving ladder.
            row["last"] = 100.0 + 0.013 * (seq - 1) + location
        rows.append(row)
        index.append((_LABEL_TIME, "110001.SH", seq))
    if include_nonphysical_label:
        row = {
            "trade_time": pd.Timestamp("2026-08-06 14:29:00.000001", tz=timezone),
            "num_trades": 0.0,
        }
        row.update(_book_fields(0, 0.0))
        rows.append(row)
        index.append((_LABEL_TIME, "127001.SZ", 0))
    panel = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["dt", "code", "seq"]),
    )
    panel.attrs["__build_day__"] = _SCORE_DAY.date().isoformat()
    return panel


def _python_golden(
    panel: pd.DataFrame,
    specs: tuple[FactorSpec, ...],
    *,
    daily_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    kernels = {
        "factor_mining_quote_execution_dynamics_v1": FactorMiningQuoteExecutionDynamicsV1,
        "factor_mining_orderbook_repricing_v1": FactorMiningOrderbookRepricingV1,
        "factor_mining_daily_expansion_v1": FactorMiningDailyExpansionV1,
    }
    values: list[pd.Series] = []
    for spec in specs:
        factor = kernels[spec.factor](output_col=spec.name)
        value = factor.compute(
            FactorComputeContext(
                panel=panel.copy(),
                daily_data={key: value.copy() for key, value in (daily_data or {}).items()},
                params=dict(spec.params or {}),
            )
        )
        values.append(value.rename(spec.name))
    return pd.concat(values, axis=1).sort_index()


def _shadow(
    module: object,
    panel: pd.DataFrame,
    specs: tuple[FactorSpec, ...],
    *,
    daily_data: dict[str, pd.DataFrame] | None = None,
    compute_backend_params: dict[str, object] | None = None,
) -> pd.DataFrame:
    return build_factor_frame_rust_legacy_reference(
        panel,
        specs,
        rust_module=module,
        daily_data=daily_data,
        compute_backend_params=compute_backend_params,
    )


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


def _daily_base_for_mixed_dispatch() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=_SCORE_DAY - pd.Timedelta(days=1), periods=24)
    rows: list[dict[str, object]] = []
    for position, trade_date in enumerate(dates):
        level = float(position + 1)
        rows.append(
            {
                "trade_date": trade_date,
                "code": "110001",
                "exchange_code": "XSHG",
                "bond_prem_ratio": 0.08 + 0.001 * level,
                "ytm": 0.02 + 0.00005 * level,
                "redemption_prem_ratio": 0.005 * level,
                "pure_redemption_value": 100.0 + level,
            }
        )
    return {"market_cbond.daily_base": pd.DataFrame(rows)}


def test_intraday_shadow_matches_python_and_keeps_per_spec_key_unions() -> None:
    module = _shadow_module()
    panel = _panel()
    specs = (*_QED_SPECS, _LRD_SPEC)
    expected = _python_golden(panel, specs)
    actual = _shadow(module, panel, specs)

    _assert_exact_shadow_frame(actual, expected)
    assert (_LABEL_TIME, "127001.SZ") in expected.index
    assert expected.loc[(_LABEL_TIME, "127001.SZ"), list(_QED_SPECS[i].name for i in range(3))].isna().all()
    assert pd.isna(expected.loc[(_LABEL_TIME, "127001.SZ"), _LRD_SPEC.name])
    assert np.isfinite(expected.loc[(_LABEL_TIME, "110001.SH")]).all()


def test_intraday_shadow_mixes_daily_and_intraday_spec_key_contracts() -> None:
    module = _shadow_module()
    panel = _panel()
    daily_spec = FactorSpec(
        name="dredemption_bondpremium_interaction",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dredemption_bondpremium_interaction"},
    )
    specs = (*_QED_SPECS, _LRD_SPEC, daily_spec)
    daily_data = _daily_base_for_mixed_dispatch()
    expected = _python_golden(panel, specs, daily_data=daily_data)
    actual = _shadow(module, panel, specs, daily_data=daily_data)
    _assert_exact_shadow_frame(actual, expected)


@pytest.mark.parametrize("seed", range(1, 21))
def test_intraday_shadow_random_seeds_are_bit_exact(seed: int) -> None:
    module = _shadow_module()
    panel = _panel(seed)
    specs = (*_QED_SPECS, _LRD_SPEC)
    _assert_exact_shadow_frame(_shadow(module, panel, specs), _python_golden(panel, specs))


@pytest.mark.parametrize("missing", ("trade_time", "last", "ask_price1", "bid_price1", "num_trades"))
def test_qed_missing_each_required_column_is_labelled_all_nan(missing: str) -> None:
    module = _shadow_module()
    panel = _panel().drop(columns=[missing])
    expected = _python_golden(panel, _QED_SPECS)
    actual = _shadow(module, panel, _QED_SPECS)
    _assert_exact_shadow_frame(actual, expected)
    assert expected.isna().all().all()


@pytest.mark.parametrize("missing", ("trade_time", "ask_price4", "bid_price2", "ask_volume5", "bid_volume3"))
def test_lrd_missing_required_book_field_raises_key_error(missing: str) -> None:
    module = _shadow_module()
    panel = _panel().drop(columns=[missing])
    with pytest.raises(KeyError):
        _python_golden(panel, (_LRD_SPEC,))
    with pytest.raises(KeyError):
        _shadow(module, panel, (_LRD_SPEC,))


@pytest.mark.parametrize("score_day_case", ("invalid_attr", "missing_attr_multi_date"))
def test_lrd_trade_time_schema_error_precedes_score_day_resolution(score_day_case: str) -> None:
    """Mirror `_score_day_frame`: ensure_trade_time runs before signal-date lookup."""

    module = _shadow_module()
    panel = _panel().drop(columns=["trade_time"])
    if score_day_case == "invalid_attr":
        panel.attrs["__build_day__"] = "not-a-date"
    else:
        rows = panel.reset_index()
        extra = rows.iloc[[0]].copy()
        extra["dt"] = _SCORE_DAY + pd.Timedelta(days=1)
        panel = pd.concat([rows, extra], ignore_index=True).set_index(["dt", "code", "seq"])
        panel.attrs.clear()

    with pytest.raises(KeyError, match="trade_time"):
        _python_golden(panel, (_LRD_SPEC,))
    with pytest.raises(KeyError, match="trade_time"):
        _shadow(module, panel, (_LRD_SPEC,))


def test_intraday_shadow_preserves_timezone_local_wall_clock_and_backend_conflict() -> None:
    module = _shadow_module()
    panel = _panel(timezone="Asia/Shanghai")
    specs = (*_QED_SPECS, _LRD_SPEC)
    expected = _python_golden(panel, specs)
    # This daily-only compatibility override is deliberately ignored by the
    # QED/LRD Python contracts: attrs[__build_day__] remains authoritative.
    actual = _shadow(
        module,
        panel,
        specs,
        compute_backend_params={"__factor_score_date": "2026-08-07"},
    )
    _assert_exact_shadow_frame(actual, expected)


def test_intraday_shadow_preserves_timezone_sensitive_labelled_score_day_match() -> None:
    module = _shadow_module()
    panel = _panel()
    # Pandas does not equate a tz-aware target day with this naive label day.
    # QED therefore has an empty labelled output; LRD treats it as the stricter
    # no-indexed-score-day error rather than a date-string match.
    panel.attrs["__build_day__"] = pd.Timestamp("2026-08-06", tz="Asia/Shanghai")
    expected = _python_golden(panel, _QED_SPECS)
    assert expected.empty
    _assert_exact_shadow_frame(_shadow(module, panel, _QED_SPECS), expected)
    with pytest.raises(ValueError, match="no indexed rows"):
        _python_golden(panel, (_LRD_SPEC,))
    with pytest.raises(ValueError, match="no indexed rows"):
        _shadow(module, panel, (_LRD_SPEC,))


def test_qed_cross_lunch_adjacency_and_duplicate_timestamp_match_python() -> None:
    module = _shadow_module()
    panel = _panel(include_nonphysical_label=False)
    rows = panel.reset_index()
    morning = [pd.Timestamp("2026-08-06 11:29:55") + pd.Timedelta(seconds=i) for i in range(6)]
    afternoon = [pd.Timestamp("2026-08-06 13:00:00") + pd.Timedelta(seconds=i) for i in range(6)]
    rows.loc[:5, "trade_time"] = morning
    rows.loc[6:11, "trade_time"] = afternoon
    rows.loc[6, "last"] = 110.0
    rows.loc[6, "num_trades"] = rows.loc[5, "num_trades"] + 2.0
    panel = rows.set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = _SCORE_DAY.date().isoformat()
    _assert_exact_shadow_frame(
        _shadow(module, panel, _QED_SPECS), _python_golden(panel, _QED_SPECS)
    )

    duplicate = panel.copy()
    duplicate.loc[(_LABEL_TIME, "110001.SH", 7), "trade_time"] = duplicate.loc[
        (_LABEL_TIME, "110001.SH", 6), "trade_time"
    ]
    expected = _python_golden(duplicate, _QED_SPECS)
    _assert_exact_shadow_frame(_shadow(module, duplicate, _QED_SPECS), expected)
    assert expected.isna().all().all()


def test_lrd_allows_duplicate_time_but_gates_lunch_and_nonfinite_log() -> None:
    module = _shadow_module()
    panel = _panel(include_nonphysical_label=False)
    duplicate = panel.copy()
    duplicate.loc[(_LABEL_TIME, "110001.SH", 5), "trade_time"] = duplicate.loc[
        (_LABEL_TIME, "110001.SH", 4), "trade_time"
    ]
    expected = _python_golden(duplicate, (_LRD_SPEC,))
    _assert_exact_shadow_frame(_shadow(module, duplicate, (_LRD_SPEC,)), expected)
    assert np.isfinite(expected.iloc[0, 0])

    rows = panel.reset_index()
    for row_index in range(6):
        for level in range(1, 6):
            rows.loc[row_index, f"bid_price{level}"] = 1.0e-308 * (1.0 + 0.001 * row_index)
            rows.loc[row_index, f"ask_price{level}"] = 2.0e-308 * (1.0 + 0.001 * row_index)
    for row_index in range(6, 12):
        for level in range(1, 6):
            rows.loc[row_index, f"bid_price{level}"] = 8.0e307 * (1.0 + 0.001 * row_index)
            rows.loc[row_index, f"ask_price{level}"] = 9.0e307 * (1.0 + 0.001 * row_index)
    rows.loc[:5, "trade_time"] = [
        pd.Timestamp("2026-08-06 11:29:55") + pd.Timedelta(seconds=i) for i in range(6)
    ]
    rows.loc[6:11, "trade_time"] = [
        pd.Timestamp("2026-08-06 13:00:00") + pd.Timedelta(seconds=i) for i in range(6)
    ]
    nonfinite = rows.set_index(["dt", "code", "seq"])
    nonfinite.attrs["__build_day__"] = _SCORE_DAY.date().isoformat()
    expected = _python_golden(nonfinite, (_LRD_SPEC,))
    _assert_exact_shadow_frame(_shadow(module, nonfinite, (_LRD_SPEC,)), expected)
    assert expected.isna().all().all()


def test_intraday_submicrosecond_cutoff_and_lrd_no_indexed_score_day() -> None:
    module = _shadow_module()
    panel = _panel(include_nonphysical_label=False)
    panel["trade_time"] = panel["trade_time"].astype("datetime64[ns]")
    inside = panel.copy()
    inside.loc[(_LABEL_TIME, "110001.SH", 11), "trade_time"] = pd.Timestamp(
        "2026-08-06 14:29:00.000000999"
    )
    specs = (*_QED_SPECS, _LRD_SPEC)
    _assert_exact_shadow_frame(_shadow(module, inside, specs), _python_golden(inside, specs))

    outside = panel.copy()
    outside.loc[(_LABEL_TIME, "110001.SH", 11), "trade_time"] = pd.Timestamp(
        "2026-08-06 14:29:00.000001000"
    )
    expected = _python_golden(outside, specs)
    _assert_exact_shadow_frame(_shadow(module, outside, specs), expected)
    assert expected.isna().all().all()

    future = panel.copy()
    future.attrs["__build_day__"] = "2026-08-07"
    with pytest.raises(ValueError, match="no indexed rows"):
        _python_golden(future, (_LRD_SPEC,))
    with pytest.raises(ValueError, match="no indexed rows"):
        _shadow(module, future, (_LRD_SPEC,))


def test_invalid_intraday_build_day_fails_closed_like_python() -> None:
    module = _shadow_module()
    panel = _panel()
    panel.attrs["__build_day__"] = "not-a-date"
    with pytest.raises(ValueError):
        _python_golden(panel, _QED_SPECS)
    with pytest.raises(ValueError):
        _shadow(module, panel, _QED_SPECS)
