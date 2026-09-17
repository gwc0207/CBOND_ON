from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_daily_contract_stock_v1 as contract_stock
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"{110000 + index:06d}.SH" for index in range(18)) + tuple(
    f"{127000 + index:06d}.SZ" for index in range(18)
)


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _panel(codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"dt": SCORE, "code": code, "seq": sequence}
            for code in codes
            for sequence in range(2)
        ]
    ).set_index(["dt", "code", "seq"])
    frame.attrs["__build_day__"] = SCORE.date().isoformat()
    return frame


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=70)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    rating_scale = ("AAA", "AA+", "AA", "AA-", "A+", "A", "A-")
    for code_index, code in enumerate(CODES):
        cb_previous = 100.0 + 1.3 * code_index
        stock_previous = 10.0 + 0.45 * code_index
        base_rating = max(0, min(len(rating_scale) - 1, 1 + (code_index % 5)))
        for day_index, day in enumerate(days):
            phase = float(day_index + 2 * code_index)
            cb_return = 0.0018 * np.sin(phase / 3.1) + 0.0007 * np.cos(phase / 5.7)
            stock_return = 0.0026 * np.cos(phase / 4.2) + 0.0009 * np.sin(phase / 2.7)
            cb_close = cb_previous * (1.0 + cb_return)
            stock_close = stock_previous * (1.0 + stock_return)
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "close_price": cb_close,
                }
            )

            # These are underlying-stock contract terms, so the terminal
            # stock close remains between put and call in the test fixture.
            put = 0.70 * stock_close * (1.0 + 0.002 * np.sin(phase / 7.0))
            call = 1.32 * stock_close * (1.0 + 0.002 * np.cos(phase / 8.0))
            conv = 0.98 * stock_close * (1.0 + 0.004 * (day_index >= 24 and code_index % 3 == 0))
            if day_index >= 46 and code_index % 4 == 0:
                put *= 1.025
            if day_index >= 39 and code_index % 5 == 0:
                call *= 0.975
            revised_trigger = 1.12 * stock_close * (1.0 + 0.006 * (day_index in {20, 49}))
            original_mode = int((code_index + day_index // 35) % 2)
            revised_mode = int((code_index + day_index // 23) % 2)
            reach = 15.0 + float(code_index % 2)
            revised_reach = reach + float((code_index + day_index // 25) % 2)
            cumulative = float((day_index + code_index) % int(reach))
            revised_cumulative = float((day_index + 2 * code_index) % int(revised_reach))
            rating_index = max(0, min(len(rating_scale) - 1, base_rating + (1 if day_index >= 34 and code_index % 6 == 0 else 0) - (1 if day_index >= 51 and code_index % 7 == 0 else 0)))
            cb_amount = 4_000_000.0 * (1.0 + 0.018 * code_index + 0.12 * np.sin(phase / 4.0))
            stk_amount = 12_000_000.0 * (1.0 + 0.012 * code_index + 0.10 * np.cos(phase / 5.0))
            cb_deal = 500.0 + 13.0 * code_index + 6.0 * day_index + 18.0 * np.sin(phase / 5.0)
            stk_deal = 1_300.0 + 19.0 * code_index + 8.0 * day_index + 23.0 * np.cos(phase / 6.0)
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "cb_conv_price": conv,
                    "cb_put_price": put,
                    "cb_call_price": call,
                    "trigger_price_revise": revised_trigger,
                    "stock_close_price": stock_close,
                    "cb_prev_close_price": cb_previous,
                    "cb_close_price": cb_close,
                    "stk_prev_close_price": stock_previous,
                    "stk_close_price": stock_close,
                    "cb_amount": cb_amount,
                    "stk_amount": stk_amount,
                    "cb_deal": cb_deal,
                    "stk_deal": stk_deal,
                    "stock_volatility": 0.020 + 0.0015 * (code_index % 6) + 0.002 * abs(np.sin(phase / 6.0)),
                    "trigger_is_price": str(original_mode),
                    "trigger_is_price_revise": str(revised_mode),
                    "trigger_cum_days": cumulative,
                    "trigger_reach_days": reach,
                    "trigger_cum_days_revise": revised_cumulative,
                    "trigger_reach_days_revise": revised_reach,
                    "in_trigger_process": 1.0 if cumulative >= reach - 2 else -1.0,
                    "rating": rating_scale[rating_index],
                    "ytm": 0.012 + 0.00016 * code_index + 0.00007 * day_index + 0.0006 * np.cos(phase / 5.0),
                }
            )
            cb_previous = cb_close
            stock_previous = stock_close
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=contract_stock.KERNEL_NAME, params={"signal": entry.signal})
        for entry in contract_stock.daily_contract_stock_catalog()
    ]


def _contaminate(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        future = terminal.copy()
        terminal["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column in {"trade_date", "code", "exchange_code"}:
                continue
            if pd.api.types.is_numeric_dtype(frame[column]):
                terminal[column] = 1_000_000.0
                future[column] = 2_000_000.0
            else:
                terminal[column] = "CCC"
                future[column] = "CC"
        out[source] = pd.concat([frame, terminal, future], ignore_index=True)
    return out


def test_catalogue_has_seven_import_only_families_and_46_signals() -> None:
    entries = contract_stock.daily_contract_stock_catalog()

    assert len(entries) == 46
    assert len({entry.signal for entry in entries}) == 46
    assert Counter(entry.family for entry in entries) == {
        "contract_term_revision_vector": 7,
        "put_call_barrier_geometry": 6,
        "daily_bond_stock_tracking_error": 6,
        "bond_stock_liquidity_reallocation": 7,
        "stock_volatility_forecast_error": 6,
        "trigger_revision_phase_wedge": 7,
        "rating_migration_decay": 7,
    }
    assert {entry.kernel for entry in entries} == {contract_stock.KERNEL_NAME}
    assert FactorRegistry.get(contract_stock.KERNEL_NAME) is contract_stock.FactorMiningDailyContractStockV1
    assert contract_stock.FactorMiningDailyContractStockV1.__name__ not in factor_operators.__all__
    assert contract_stock.FactorMiningDailyContractStockV1.requires_stock_panel is False
    assert contract_stock.FactorMiningDailyContractStockV1.requires_bond_stock_map is False


def test_daily_requirements_keep_independent_price_anchor_and_signal_specific_base_fields() -> None:
    requirements = contract_stock.FactorMiningDailyContractStockV1.daily_requirements(
        {"signal": "barrier_center_position"}
    )

    assert [item.source for item in requirements] == [
        "market_cbond.daily_price",
        "market_cbond.daily_base",
    ]
    assert requirements[0].columns == ("exchange_code", "close_price")
    assert "stock_close_price" in requirements[1].columns
    assert "trigger_price_revise" in requirements[1].columns
    assert requirements[1].lookback_days >= 65
    all_requirements = contract_stock.FactorMiningDailyContractStockV1.daily_requirements()
    assert {item.source for item in all_requirements} == {
        "market_cbond.daily_price",
        "market_cbond.daily_base",
    }
    all_base = next(item for item in all_requirements if item.source == "market_cbond.daily_base")
    assert "base_rate" not in all_base.columns
    assert "trigger_type" not in all_base.columns
    assert "puredebt_prem_ratio" not in all_base.columns


def test_all_signals_build_without_inf_and_have_cross_sectional_support() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())

    assert frame.columns.tolist() == [entry.signal for entry in contract_stock.daily_contract_stock_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= 30).all()


def test_all_signals_ignore_score_day_and_future_daily_rows() -> None:
    baseline = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())
    contaminated = build_factor_frame(_panel(), _specs(), daily_data=_contaminate(_daily_sources()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_source_field_and_strict_prior_duplicate_fail_closed() -> None:
    kernel = contract_stock.FactorMiningDailyContractStockV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(FactorComputeContext(panel=_panel(), daily_data={}, params={"signal": "ctr_conv_term_change1"}))

    sources = _daily_sources()
    missing_stock_close = sources["market_cbond.daily_base"].drop(columns=["stock_close_price"])
    with pytest.raises(KeyError, match="stock_close_price"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={
                    "market_cbond.daily_price": sources["market_cbond.daily_price"],
                    "market_cbond.daily_base": missing_stock_close,
                },
                params={"signal": "barrier_center_position"},
            )
        )

    duplicate_price = pd.concat(
        [sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": duplicate_price},
                params={"signal": "ctr_conv_term_change1"},
            )
        )


def test_stale_base_state_on_independent_price_anchor_is_nan_for_every_family() -> None:
    sources = _daily_sources()
    stale_code = _bare(CODES[0])
    latest_day = sources["market_cbond.daily_price"]["trade_date"].max()
    base = sources["market_cbond.daily_base"]
    stale_base = base.loc[~((base["code"] == stale_code) & (base["trade_date"] == latest_day))].copy()

    frame = build_factor_frame(
        _panel(),
        _specs(),
        daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": stale_base},
    )

    assert frame.loc[(SCORE, CODES[0])].isna().all()
    assert frame.loc[(SCORE, CODES[1])].notna().any()


def test_one_session_step_does_not_silently_cross_a_missing_base_day() -> None:
    sources = _daily_sources()
    code = _bare(CODES[0])
    price = sources["market_cbond.daily_price"]
    prior_day = sorted(price.loc[price["code"] == code, "trade_date"].unique())[-2]
    base = sources["market_cbond.daily_base"]
    gapped_base = base.loc[~((base["code"] == code) & (base["trade_date"] == prior_day))].copy()
    kernel = contract_stock.FactorMiningDailyContractStockV1()

    out = kernel.compute(
        FactorComputeContext(
            panel=_panel(),
            daily_data={"market_cbond.daily_price": price, "market_cbond.daily_base": gapped_base},
            params={"signal": "ctr_conv_term_change1"},
        )
    )

    assert np.isnan(out.loc[(SCORE, CODES[0])])
    assert np.isfinite(out.loc[(SCORE, CODES[1])])


def test_explicit_rating_mapping_and_representative_family_formulas() -> None:
    assert contract_stock._rating_value("AAA") > contract_stock._rating_value("AA+") > contract_stock._rating_value("AA")
    assert contract_stock._rating_value("BBB-") > contract_stock._rating_value("BB+")
    assert np.isnan(contract_stock._rating_value("unrated"))

    barrier = contract_stock._barrier_geometry_metrics(
        pd.DataFrame(
            {
                "stock_close_price": [15.0],
                "cb_put_price": [10.0],
                "cb_call_price": [20.0],
                "trigger_price_revise": [18.0],
            }
        )
    )
    assert barrier["barrier_center_position"] == pytest.approx(0.5)
    assert barrier["barrier_width_to_price"] == pytest.approx(10.0 / 15.0)
    assert barrier["barrier_revision_trigger_position"] == pytest.approx(0.8)

    terms = pd.DataFrame(
        {
            "cb_conv_price": [10.0, 11.0],
            "cb_put_price": [8.0, 8.0],
            "cb_call_price": [15.0, 15.0],
        }
    )
    assert contract_stock._contract_term_revision_metrics(terms)["ctr_conv_term_change1"] == pytest.approx(0.1)

    trigger = pd.DataFrame(
        {
            "trigger_is_price": ["0"] * 61,
            "trigger_is_price_revise": ["1"] * 61,
            "trigger_cum_days": np.arange(61, dtype="float64") % 15,
            "trigger_reach_days": [15.0] * 61,
            "trigger_cum_days_revise": np.arange(61, dtype="float64") % 16,
            "trigger_reach_days_revise": [16.0] * 61,
            "trigger_price_revise": [12.0] * 60 + [12.6],
            "in_trigger_process": [-1.0] * 60 + [1.0],
        }
    )
    trigger_metrics = contract_stock._trigger_revision_phase_metrics(trigger)
    assert trigger_metrics["trw_mode_wedge"] == pytest.approx(1.0)
    assert trigger_metrics["trw_required_days_wedge"] == pytest.approx(1.0 / 15.0)
    assert trigger_metrics["trw_revision_recency60"] == pytest.approx(1.0)
    assert trigger_metrics["trw_post_revision_activation"] == pytest.approx(1.0)

    ratings = pd.DataFrame(
        {
            "rating": ["AA"] * 20 + ["AA+"] + ["AA+"] * 40,
            "ytm": np.linspace(0.01, 0.02, 61),
        }
    )
    rating_metrics = contract_stock._rating_migration_metrics(ratings)
    assert rating_metrics["rating_current_ordinal"] == pytest.approx(contract_stock._RATING_ORDINAL["AA+"])
    assert rating_metrics["rating_upgrade_decay20"] >= 0.0
    assert rating_metrics["rating_event_recency60"] > 0.0
