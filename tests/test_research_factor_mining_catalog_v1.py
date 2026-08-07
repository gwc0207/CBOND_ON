from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.defs import research_factor_mining_catalog_v1 as catalog
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"


def _path_rows(
    code: str,
    *,
    start: float,
    end: float,
    cumulative_volume: list[float] | None = None,
    dt: pd.Timestamp = DT,
) -> list[dict[str, object]]:
    """Return a complete nine-tick T1430 path accepted by the catalogue."""

    count = 9
    prices = np.linspace(start, end, count)
    volumes = cumulative_volume or [float(10 * (index + 1)) for index in range(count)]
    if len(volumes) != count:
        raise ValueError("test path must contain exactly nine cumulative snapshots")
    factor_dt = pd.Timestamp(dt)
    start_time = factor_dt.normalize() + pd.Timedelta(hours=14, minutes=20)
    rows: list[dict[str, object]] = []
    for seq, (price, volume) in enumerate(zip(prices, volumes, strict=True)):
        row: dict[str, object] = {
            "dt": factor_dt,
            "code": code,
            "seq": seq,
            "trade_time": start_time + pd.Timedelta(minutes=seq),
            "pre_close": start,
            "open": start,
            "last": float(price),
            "volume": float(volume),
            "amount": float(1_000 * (seq + 1)),
            "num_trades": float(5 * (seq + 1)),
            "high_limited": float(price * 1.20),
            "low_limited": float(price * 0.80),
        }
        for level in range(1, 6):
            row[f"ask_price{level}"] = float(price + 0.01 * level)
            row[f"bid_price{level}"] = float(price - 0.01 * level)
            row[f"ask_volume{level}"] = float(20 + level + seq)
            row[f"bid_volume{level}"] = float(25 + level + seq)
        rows.append(row)
    return rows


def _panel(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index(["dt", "code", "seq"])


def _daily_price_row(
    day: pd.Timestamp,
    *,
    close: float,
    open_px: float,
    code: str = "110001",
    exchange_code: str = "SH",
) -> dict[str, object]:
    """Build one schema-complete daily_price row without relying on production data."""

    row: dict[str, object] = {column: 1.0 for column in catalog._DAILY_PRICE_COLUMNS}
    row.update(
        {
            "trade_date": pd.Timestamp(day).normalize(),
            "code": code,
            "exchange_code": exchange_code,
            "prev_close_price": 100.0,
            "act_prev_close_price": 100.0,
            "open_price": open_px,
            "close_price": close,
            "high_price": max(open_px, close) + 1.0,
            "low_price": min(open_px, close) - 1.0,
            "volume": 1_000.0,
            "amount": 100_000.0,
            "deal": 100.0,
        }
    )
    return row


def _daily_price_history(*, include_signal_and_future_outliers: bool) -> pd.DataFrame:
    prior_days = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=20)
    rows = [
        _daily_price_row(day, close=100.2 + 0.1 * index, open_px=99.8 + 0.1 * index)
        for index, day in enumerate(prior_days)
    ]
    if include_signal_and_future_outliers:
        # These values would dominate dpx_body_range if signal/future daily
        # rows leaked into the T=14:30 factor context.
        rows.extend(
            [
                _daily_price_row(DT.normalize(), close=1_000.0, open_px=100.0),
                _daily_price_row(DT.normalize() + pd.Timedelta(days=1), close=2_000.0, open_px=100.0),
            ]
        )
    return pd.DataFrame(rows)


def _daily_base_row(
    day: pd.Timestamp,
    *,
    stock_code: str,
    code: str = "110001",
    exchange_code: str = "SH",
) -> dict[str, object]:
    """Build a schema-complete daily_base row with an explicit stock mapping."""

    row: dict[str, object] = {column: 1.0 for column in catalog._DAILY_BASE_COLUMNS}
    row.update(
        {
            "trade_date": pd.Timestamp(day).normalize(),
            "code": code,
            "exchange_code": exchange_code,
            "stock_code": stock_code,
            "bond_prem_ratio": 0.20,
            "puredebt_prem_ratio": 0.10,
            "cb_close_price": 100.0,
            "conv_value": 110.0,
            "pure_redemption_value": 90.0,
            "remain_size": 1_000.0,
            "duration": 2.0,
            "stock_volatility": 0.30,
            "cb_amount": 100_000.0,
            "stk_amount": 1_000_000.0,
            "cb_deal": 100.0,
            "stk_deal": 1_000.0,
        }
    )
    return row


def _with_build_day(frame: pd.DataFrame, day: pd.Timestamp = DT) -> pd.DataFrame:
    frame.attrs["__build_day__"] = pd.Timestamp(day).date().isoformat()
    return frame


def test_catalogue_has_exact_family_first_coverage_and_registered_kernels() -> None:
    entries = catalog.factor_mining_catalog()
    expected_kernels = {
        "factor_mining_intraday_catalog_v1": catalog.FactorMiningIntradayCatalogV1,
        "factor_mining_daily_catalog_v1": catalog.FactorMiningDailyCatalogV1,
        "factor_mining_cross_asset_catalog_v1": catalog.FactorMiningCrossAssetCatalogV1,
        "factor_mining_hybrid_catalog_v1": catalog.FactorMiningHybridCatalogV1,
    }

    assert len(entries) == 238
    assert len({entry.signal for entry in entries}) == len(entries)
    assert Counter(entry.kernel for entry in entries) == {
        "factor_mining_intraday_catalog_v1": 70,
        "factor_mining_daily_catalog_v1": 105,
        "factor_mining_cross_asset_catalog_v1": 42,
        "factor_mining_hybrid_catalog_v1": 21,
    }
    assert all(entry.family and entry.hypothesis for entry in entries)
    assert {entry.kernel for entry in entries} == set(expected_kernels)
    for kernel, factor_class in expected_kernels.items():
        assert FactorRegistry.get(kernel) is factor_class
        assert not hasattr(factor_defs, factor_class.__name__)
        assert factor_class.__name__ not in factor_defs.__all__

    requirements = catalog.FactorMiningDailyCatalogV1.daily_requirements()
    assert [item.source for item in requirements] == [
        "market_cbond.daily_twap",
        "market_cbond.daily_price",
        "market_cbond.daily_base",
    ]
    assert catalog.FactorMiningCrossAssetCatalogV1.requires_stock_panel is True


def test_daily_catalogue_excludes_signal_day_and_future_daily_rows() -> None:
    panel = _panel(*_path_rows(BOND, start=100.0, end=102.0))
    baseline = catalog.FactorMiningDailyCatalogV1().compute(
        FactorComputeContext(
            panel=panel,
            daily_data={"market_cbond.daily_price": _daily_price_history(include_signal_and_future_outliers=False)},
            params={"signal": "dpx_body_range"},
        )
    )
    contaminated = catalog.FactorMiningDailyCatalogV1().compute(
        FactorComputeContext(
            panel=panel,
            daily_data={"market_cbond.daily_price": _daily_price_history(include_signal_and_future_outliers=True)},
            params={"signal": "dpx_body_range"},
        )
    )

    assert np.isfinite(baseline.iloc[0])
    pd.testing.assert_series_equal(contaminated, baseline, check_exact=True)


def test_counter_reset_stays_explicitly_missing_instead_of_becoming_zero_flow() -> None:
    good = _panel(*_path_rows(BOND, start=100.0, end=102.0))
    reset = _panel(
        *_path_rows(
            BOND,
            start=100.0,
            end=102.0,
            cumulative_volume=[10.0, 20.0, 30.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0],
        )
    )
    good_out = catalog.FactorMiningIntradayCatalogV1().compute(
        FactorComputeContext(panel=good, params={"signal": "intra_volume_entropy"})
    )
    reset_out = catalog.FactorMiningIntradayCatalogV1().compute(
        FactorComputeContext(panel=reset, params={"signal": "intra_volume_entropy"})
    )

    assert np.isfinite(good_out.loc[(DT, BOND)])
    assert np.isnan(reset_out.loc[(DT, BOND)])


def test_cross_catalogue_uses_tminus1_daily_base_mapping_not_current_or_context_map() -> None:
    bond_panel = _panel(*_path_rows(BOND, start=100.0, end=102.0))
    stock_panel = _panel(
        *_path_rows("600001.SH", start=10.0, end=11.0),
        *_path_rows("600002.SH", start=10.0, end=9.5),
    )
    # T-1 maps the bond to 600001 (10% intraday return).  The deliberately
    # conflicting signal-day row and context map both point to 600002 (-5%).
    # A same-day or ctx.bond_stock_map implementation would therefore yield
    # +7%, while the permitted T-1 mapping must yield 2% - 10% = -8%.
    daily_base = pd.DataFrame(
        [
            _daily_base_row(DT.normalize() - pd.offsets.BDay(1), stock_code="600001.SH"),
            _daily_base_row(DT.normalize(), stock_code="600002.SH"),
        ]
    )
    out = catalog.FactorMiningCrossAssetCatalogV1().compute(
        FactorComputeContext(
            panel=bond_panel,
            stock_panel=stock_panel,
            bond_stock_map=pd.DataFrame({"bond_code": [BOND], "stock_code": ["600002.SH"]}),
            daily_data={
                "market_cbond.daily_base": daily_base,
                "market_cbond.daily_price": pd.DataFrame(
                    [_daily_price_row(DT.normalize() - pd.offsets.BDay(1), close=100.0, open_px=100.0)]
                ),
            },
            params={"signal": "cross_bond_stock_return_gap"},
        )
    )

    assert out.loc[(DT, BOND)] == pytest.approx(-0.08, abs=1e-12)


def test_cross_catalogue_rejects_stale_base_mapping_behind_prior_market_anchor() -> None:
    bond_panel = _panel(*_path_rows(BOND, start=100.0, end=102.0))
    stock_panel = _panel(*_path_rows("600001.SH", start=10.0, end=11.0))
    # The T-1 daily_price row proves the prior market session is T-1, while
    # daily_base has only T-2.  Cross factors must not reuse that stale map.
    daily_base = pd.DataFrame(
        [_daily_base_row(DT.normalize() - pd.offsets.BDay(2), stock_code="600001.SH")]
    )
    daily_price = pd.DataFrame(
        [_daily_price_row(DT.normalize() - pd.offsets.BDay(1), close=100.0, open_px=100.0)]
    )
    out = catalog.FactorMiningCrossAssetCatalogV1().compute(
        FactorComputeContext(
            panel=bond_panel,
            stock_panel=stock_panel,
            daily_data={
                "market_cbond.daily_base": daily_base,
                "market_cbond.daily_price": daily_price,
            },
            params={"signal": "cross_bond_stock_return_gap"},
        )
    )

    assert np.isnan(out.loc[(DT, BOND)])


def test_catalogue_kernel_output_is_a_two_level_dt_code_multiindex_series() -> None:
    second_bond = "110002.SH"
    panel = _panel(
        *_path_rows(BOND, start=100.0, end=102.0),
        *_path_rows(second_bond, start=100.0, end=98.0),
    )
    out = catalog.FactorMiningIntradayCatalogV1().compute(
        FactorComputeContext(panel=panel, params={"signal": "intra_full_return"})
    )

    assert isinstance(out, pd.Series)
    assert isinstance(out.index, pd.MultiIndex)
    assert out.index.names == ["dt", "code"]
    assert out.index.nlevels == 2
    assert out.index.tolist() == [(DT, BOND), (DT, second_bond)]
    assert out.name == "intra_full_return"
    assert out.loc[(DT, BOND)] == pytest.approx(0.02)
    assert out.loc[(DT, second_bond)] == pytest.approx(-0.02)


def test_factor_spec_instance_name_does_not_replace_catalogue_kernel() -> None:
    """The real builder gives the factor object its instance/output name.

    Every catalogue spec deliberately has a human-readable signal name rather
    than the registered kernel name.  The kernel selector must therefore stay
    fixed on the class even after FactorSpec.build() performs that renaming.
    """

    spec = FactorSpec(
        name="intra_full_return",
        factor="factor_mining_intraday_catalog_v1",
        params={"signal": "intra_full_return"},
    )
    assert spec.build().name == "intra_full_return"

    frame = build_factor_frame(
        _panel(*_path_rows(BOND, start=100.0, end=102.0)),
        [spec],
    )

    assert frame.columns.tolist() == ["intra_full_return"]
    assert frame.loc[(DT, BOND), "intra_full_return"] == pytest.approx(0.02)


def test_catalogue_caches_validated_output_index_across_concrete_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _panel(*_path_rows(BOND, start=100.0, end=102.0))
    calls = 0
    original = catalog._signal_day_panel

    def counted(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(catalog, "_signal_day_panel", counted)
    specs = [
        FactorSpec(
            name="intra_full_return",
            factor="factor_mining_intraday_catalog_v1",
            params={"signal": "intra_full_return"},
        ),
        FactorSpec(
            name="intra_path_efficiency",
            factor="factor_mining_intraday_catalog_v1",
            params={"signal": "intra_path_efficiency"},
        ),
    ]

    frame = build_factor_frame(panel, specs)

    assert frame.columns.tolist() == ["intra_full_return", "intra_path_efficiency"]
    # One validated output-index build plus one shared intraday feature build;
    # neither concrete spec may rescan the rolling panel independently.
    assert calls == 2


def test_daily_catalogue_keeps_exchange_when_bare_instrument_codes_collide() -> None:
    """DataHub daily context carries instrument_code as bare code plus exchange."""

    prior_days = pd.bdate_range(end=DT.normalize() - pd.offsets.BDay(1), periods=20)
    rows: list[dict[str, object]] = []
    for day in prior_days:
        # Both rows have the same instrument_code and trade_date.  SZ has a
        # +0.5 candle-body ratio and SH has -0.5, so bare-code grouping would
        # either raise a false duplicate error or select the wrong history.
        rows.append(
            _daily_price_row(
                day,
                close=102.0,
                open_px=100.0,
                code="127017",
                exchange_code="SZ",
            )
        )
        rows.append(
            _daily_price_row(
                day,
                close=100.0,
                open_px=102.0,
                code="127017",
                exchange_code="SH",
            )
        )
    requirements = catalog.FactorMiningDailyCatalogV1.daily_requirements()
    price_requirement = next(item for item in requirements if item.source == "market_cbond.daily_price")
    assert "exchange_code" in price_requirement.columns

    out = catalog.FactorMiningDailyCatalogV1().compute(
        FactorComputeContext(
            panel=_panel(*_path_rows("127017.SZ", start=100.0, end=102.0)),
            daily_data={"market_cbond.daily_price": pd.DataFrame(rows)},
            params={"signal": "dpx_body_range"},
        )
    )

    assert out.index.tolist() == [(DT, "127017.SZ")]
    assert out.iloc[0] == pytest.approx(0.5)


def test_catalogue_restricts_mixed_panel_to_build_day_and_ignores_prior_values() -> None:
    previous_dt = DT - pd.offsets.BDay(1)
    baseline_panel = _with_build_day(
        _panel(
            *_path_rows(BOND, start=100.0, end=150.0, dt=previous_dt),
            *_path_rows(BOND, start=100.0, end=102.0, dt=DT),
        )
    )
    perturbed_panel = _with_build_day(
        _panel(
            *_path_rows(BOND, start=100.0, end=50.0, dt=previous_dt),
            *_path_rows(BOND, start=100.0, end=102.0, dt=DT),
        )
    )
    baseline = catalog.FactorMiningIntradayCatalogV1().compute(
        FactorComputeContext(panel=baseline_panel, params={"signal": "intra_full_return"})
    )
    perturbed = catalog.FactorMiningIntradayCatalogV1().compute(
        FactorComputeContext(panel=perturbed_panel, params={"signal": "intra_full_return"})
    )

    assert baseline.index.tolist() == [(DT, BOND)]
    assert baseline.iloc[0] == pytest.approx(0.02)
    pd.testing.assert_series_equal(perturbed, baseline, check_exact=True)


def _cross_context_with_prior_panel_rows(
    *,
    previous_bond_end: float,
    previous_stock_end: float,
) -> FactorComputeContext:
    previous_dt = DT - pd.offsets.BDay(1)
    bond_panel = _with_build_day(
        _panel(
            *_path_rows(BOND, start=100.0, end=previous_bond_end, dt=previous_dt),
            *_path_rows(BOND, start=100.0, end=102.0, dt=DT),
        )
    )
    stock_panel = _with_build_day(
        _panel(
            *_path_rows("600001.SH", start=10.0, end=previous_stock_end, dt=previous_dt),
            *_path_rows("600001.SH", start=10.0, end=11.0, dt=DT),
        )
    )
    return FactorComputeContext(
        panel=bond_panel,
        stock_panel=stock_panel,
        daily_data={
            "market_cbond.daily_base": pd.DataFrame(
                [_daily_base_row(DT.normalize() - pd.offsets.BDay(1), stock_code="600001.SH")]
            )
        },
        params={"signal": "cross_bond_stock_return_gap"},
    )


def test_cross_catalogue_restricts_bond_and_stock_panels_to_build_day() -> None:
    baseline = catalog.FactorMiningCrossAssetCatalogV1().compute(
        _cross_context_with_prior_panel_rows(previous_bond_end=160.0, previous_stock_end=5.0)
    )
    perturbed = catalog.FactorMiningCrossAssetCatalogV1().compute(
        _cross_context_with_prior_panel_rows(previous_bond_end=50.0, previous_stock_end=20.0)
    )

    assert baseline.index.tolist() == [(DT, BOND)]
    assert baseline.iloc[0] == pytest.approx(-0.08, abs=1e-12)
    pd.testing.assert_series_equal(perturbed, baseline, check_exact=True)


def _prior_trade_time_rows_relabelled_to_score_day(
    code: str,
    *,
    start: float,
    end: float,
) -> list[dict[str, object]]:
    """Emulate clean_direct's rolling history with target-day index labels."""

    prior_day = DT - pd.offsets.BDay(1)
    rows = _path_rows(code, start=start, end=end, dt=prior_day)
    for row in rows:
        row["dt"] = DT
        row["seq"] = int(row["seq"]) + 100
    return rows


def test_intraday_catalogue_rejects_prior_trade_times_with_score_day_dt_label() -> None:
    current_rows = _path_rows(BOND, start=100.0, end=102.0)
    baseline = catalog.FactorMiningIntradayCatalogV1().compute(
        FactorComputeContext(
            panel=_with_build_day(_panel(*current_rows)),
            params={"signal": "intra_full_return"},
        )
    )
    contaminated = catalog.FactorMiningIntradayCatalogV1().compute(
        FactorComputeContext(
            panel=_with_build_day(
                _panel(
                    *_prior_trade_time_rows_relabelled_to_score_day(BOND, start=50.0, end=100.0),
                    *current_rows,
                )
            ),
            params={"signal": "intra_full_return"},
        )
    )

    assert baseline.iloc[0] == pytest.approx(0.02)
    pd.testing.assert_series_equal(contaminated, baseline, check_exact=True)


def test_cross_catalogue_rejects_prior_trade_times_with_score_day_dt_label() -> None:
    bond_rows = _path_rows(BOND, start=100.0, end=102.0)
    stock_rows = _path_rows("600001.SH", start=10.0, end=11.0)
    daily_data = {
        "market_cbond.daily_base": pd.DataFrame(
            [_daily_base_row(DT.normalize() - pd.offsets.BDay(1), stock_code="600001.SH")]
        ),
        "market_cbond.daily_price": pd.DataFrame(
            [_daily_price_row(DT.normalize() - pd.offsets.BDay(1), close=100.0, open_px=100.0)]
        ),
    }
    baseline = catalog.FactorMiningCrossAssetCatalogV1().compute(
        FactorComputeContext(
            panel=_with_build_day(_panel(*bond_rows)),
            stock_panel=_with_build_day(_panel(*stock_rows)),
            daily_data=daily_data,
            params={"signal": "cross_stock_early_bond_late_response"},
        )
    )
    contaminated = catalog.FactorMiningCrossAssetCatalogV1().compute(
        FactorComputeContext(
            panel=_with_build_day(
                _panel(
                    *_prior_trade_time_rows_relabelled_to_score_day(BOND, start=50.0, end=100.0),
                    *bond_rows,
                )
            ),
            stock_panel=_with_build_day(
                _panel(
                    *_prior_trade_time_rows_relabelled_to_score_day("600001.SH", start=5.0, end=10.0),
                    *stock_rows,
                )
            ),
            daily_data=daily_data,
            params={"signal": "cross_stock_early_bond_late_response"},
        )
    )

    assert baseline.notna().all()
    pd.testing.assert_series_equal(contaminated, baseline, check_exact=True)
