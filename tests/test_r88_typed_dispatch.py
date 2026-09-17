from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.infra.factors import rust_backend


def _isolated_module() -> object:
    raw = os.environ.get("CBOND_ON_R88_TYPED_SITE", "").strip()
    if not raw:
        pytest.skip("set CBOND_ON_R88_TYPED_SITE to run against an isolated Rust wheel")
    site = Path(raw).resolve()
    if not (site / "cbond_on_rust").is_dir():
        raise AssertionError(f"isolated Rust wheel is absent: {site}")
    for name in tuple(sys.modules):
        if name == "cbond_on_rust" or name.startswith("cbond_on_rust."):
            del sys.modules[name]
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    module = importlib.import_module("cbond_on_rust")
    extension = importlib.import_module("cbond_on_rust.cbond_on_rust")
    assert Path(extension.__file__).resolve().is_relative_to(site)
    return module


def _panel(rows: list[dict], *, day: str = "2026-08-06") -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(f"{day} 14:30:00"), "110001.SH", index) for index in range(len(rows))],
        names=["dt", "code", "seq"],
    )
    frame = pd.DataFrame(rows, index=index)
    frame.attrs["__build_day__"] = day
    return frame


def _direct_spec() -> FactorSpec:
    return FactorSpec(
        name="exp_rotation_segment_return_dispersion",
        factor="factor_mining_intraday_expansion_v1",
        params={
            "signal": "exp_rotation_segment_return_dispersion",
            "family": "clock_time_rotation",
        },
    )


def test_common_compute_route_enforces_r88_physical_1429_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_module()
    monkeypatch.setattr(rust_backend, "_import_rust_module", lambda: module)
    rows = [
        {"trade_time": f"2026-08-06 {clock}", "last": last}
        for clock, last in [
            ("09:30:00", 100.0),
            ("09:31:00", 101.0),
            ("10:30:00", 102.0),
            ("10:31:00", 104.0),
            ("13:30:00", 105.0),
            ("13:31:00", 108.0),
            ("14:30:00", 1.0),
        ]
    ]
    baseline = rust_backend.build_factor_frame_rust(_panel(rows), [_direct_spec()])
    contaminated = rows.copy()
    contaminated[-1] = {"trade_time": "2026-08-06 14:30:00", "last": 9_999_999.0}
    after_cutoff = rust_backend.build_factor_frame_rust(_panel(contaminated), [_direct_spec()])

    assert baseline.iloc[0, 0] > 0.0
    assert baseline.iloc[0, 0] == after_cutoff.iloc[0, 0]


def _beta_sources(score_day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    days = pd.date_range(end=score_day - pd.Timedelta(days=1), periods=60, freq="D")
    stock_return = (np.arange(60, dtype="float64") - 29.5) * 0.001
    price = pd.DataFrame(
        {
            "trade_date": days,
            "code": "110001",
            "exchange_code": "XSHG",
            "prev_close_price": 100.0,
            "close_price": 100.0 * np.exp(2.0 * stock_return),
        }
    )
    base = pd.DataFrame(
        {
            "trade_date": days,
            "code": "110001",
            "exchange_code": "XSHG",
            "stk_prev_close_price": 50.0,
            "stk_close_price": 50.0 * np.exp(stock_return),
        }
    )
    return {
        "market_cbond.daily_price": price,
        "market_cbond.daily_base": base,
    }


def test_common_compute_route_keeps_r88_daily_strict_tminus1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_module()
    monkeypatch.setattr(rust_backend, "_import_rust_module", lambda: module)
    score_day = pd.Timestamp("2026-08-06")
    spec = FactorSpec(
        name="bsab_upside_beta60",
        factor="factor_mining_daily_asymmetric_equity_beta_v1",
        params={
            "signal": "bsab_upside_beta60",
            "family": "prior_asymmetric_equity_beta",
        },
    )
    baseline_sources = _beta_sources(score_day)
    contaminated_sources = {key: value.copy() for key, value in baseline_sources.items()}
    contaminated_sources["market_cbond.daily_price"] = pd.concat(
        [
            contaminated_sources["market_cbond.daily_price"],
            pd.DataFrame(
                {
                    "trade_date": [score_day],
                    "code": ["110001"],
                    "exchange_code": ["XSHG"],
                    "prev_close_price": [100.0],
                    "close_price": [9_999_999.0],
                }
            ),
        ],
        ignore_index=True,
    )
    contaminated_sources["market_cbond.daily_base"] = pd.concat(
        [
            contaminated_sources["market_cbond.daily_base"],
            pd.DataFrame(
                {
                    "trade_date": [score_day],
                    "code": ["110001"],
                    "exchange_code": ["XSHG"],
                    "stk_prev_close_price": [50.0],
                    "stk_close_price": [9_999_999.0],
                }
            ),
        ],
        ignore_index=True,
    )
    panel = _panel([{"trade_time": "2026-08-06 14:29:00", "last": 100.0}])
    baseline = rust_backend.build_factor_frame_rust(panel, [spec], daily_data=baseline_sources)
    contaminated = rust_backend.build_factor_frame_rust(panel, [spec], daily_data=contaminated_sources)

    assert baseline.iloc[0, 0] == pytest.approx(2.0, abs=1e-12)
    assert baseline.iloc[0, 0] == contaminated.iloc[0, 0]


def test_remaining_itr_forwards_only_required_stock_and_map_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_module()
    monkeypatch.setattr(rust_backend, "_import_rust_module", lambda: module)
    clock = pd.date_range("2026-08-06 09:30:00", periods=14, freq="5min")
    prices = 100.0 * np.exp(np.linspace(0.0, 0.04, len(clock)))
    bond = _panel(
        [{"trade_time": time, "last": price} for time, price in zip(clock, prices, strict=True)]
    )
    stock_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-08-06 14:30:00"), "600000.SH", index) for index in range(len(clock))],
        names=["dt", "code", "seq"],
    )
    stock = pd.DataFrame(
        {"trade_time": clock, "last": prices / 2.0},
        index=stock_index,
    )
    stock.attrs["__build_day__"] = "2026-08-06"
    mapping = pd.DataFrame(
        {"code": ["110001.SH"], "stock_code": ["600000.SH"], "trade_date": ["2026-08-06"]}
    )
    spec = FactorSpec(
        name="itr_stock_shock_same_bin_directional_agreement",
        factor="factor_mining_intraday_transmission_response_v1",
        params={
            "signal": "itr_stock_shock_same_bin_directional_agreement",
            "family": "intraday_stock_shock_directional_response",
        },
    )
    out = rust_backend.build_factor_frame_rust(
        bond,
        [spec],
        stock_panel=stock,
        bond_stock_map=mapping,
    )

    assert list(out.columns) == [spec.name]
    assert out.index.names == ["dt", "code"]
    assert len(out) == 1
    assert out.iloc[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_r88_family_mismatch_fails_before_factor_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_module()
    monkeypatch.setattr(rust_backend, "_import_rust_module", lambda: module)
    invalid = FactorSpec(
        name="bad_family",
        factor="factor_mining_intraday_expansion_v1",
        params={
            "signal": "exp_noise_variance_ratio_2",
            "family": "clock_time_rotation",
        },
    )
    with pytest.raises(ValueError, match="requires params.family"):
        rust_backend.build_factor_frame_rust(
            _panel([{"trade_time": "2026-08-06 14:29:00", "last": 100.0}]),
            [invalid],
        )
