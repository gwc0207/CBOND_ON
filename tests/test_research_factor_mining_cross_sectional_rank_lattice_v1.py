from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_cross_sectional_rank_lattice_v1 as lattice
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"110{index:03d}.SH" for index in range(24)) + tuple(
    f"127{index:03d}.SZ" for index in range(24)
)


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _panel(codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    panel = pd.DataFrame(
        [{"dt": SCORE, "code": code, "seq": seq} for code in codes for seq in range(2)]
    ).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = SCORE.date().isoformat()
    return panel


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=8)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        for day_index, day in enumerate(days):
            phase = float(day_index + 1.9 * code_index)
            close = 95.0 + 0.8 * code_index + 0.35 * np.sin(phase / 3.0)
            stock_close = 8.0 + 0.19 * code_index + 0.08 * np.cos(phase / 4.0)
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "close_price": close,
                }
            )
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": _exchange(code),
                    "bond_prem_ratio": 0.04 + 0.0016 * code_index + 0.005 * np.sin(phase / 5.0),
                    "conv_value": close * (0.72 + 0.008 * (code_index % 17) + 0.01 * np.cos(phase / 4.0)),
                    "stock_volatility": 0.12 + 0.003 * (code_index % 13) + 0.009 * np.sin(phase / 2.7),
                    "duration": 1.0 + 0.09 * (code_index % 19) + 0.05 * np.cos(phase / 6.0),
                    "debt_puredebt_ratio": 0.91 + 0.006 * (code_index % 15) + 0.002 * np.sin(phase / 3.2),
                    "puredebt_prem_ratio": 0.04 + 0.003 * (code_index % 11) + 0.002 * np.cos(phase / 4.1),
                    "pure_redemption_value": close * (0.84 + 0.004 * (code_index % 12)),
                    "redemption_prem_ratio": 0.015 + 0.002 * (code_index % 16) + 0.003 * np.sin(phase / 4.8),
                    "ytm": 0.011 + 0.00035 * (code_index % 17) + 0.0002 * np.cos(phase / 3.5),
                    "turnover_rate": 0.003 + 0.0004 * (code_index % 18) + 0.0003 * np.sin(phase / 2.9),
                    "remain_size": 0.55e9 + 0.06e9 * code_index + 0.04e9 * np.cos(phase / 5.3),
                    "cb_amount": 5.0e7 + 1.1e6 * code_index + 3.0e6 * np.sin(phase / 3.7),
                    "stk_amount": 2.0e8 + 2.7e6 * code_index + 7.0e6 * np.cos(phase / 4.3),
                    "cb_deal": 800.0 + 17.0 * code_index + 9.0 * np.sin(phase / 5.1),
                    "cb_conv_price": stock_close * (0.78 + 0.01 * (code_index % 9)),
                    "cb_put_price": stock_close * (0.86 + 0.009 * (code_index % 10)),
                    "cb_call_price": stock_close * (1.11 + 0.008 * (code_index % 12)),
                    "trigger_price_revise": stock_close * (0.92 + 0.011 * (code_index % 14)),
                    "stock_close_price": stock_close,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=lattice.KERNEL_NAME, params={"signal": entry.signal})
        for entry in lattice.cross_sectional_rank_lattice_catalog()
    ]


def _contaminate(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        score = terminal.copy()
        future = terminal.copy()
        score["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column not in {"trade_date", "code", "exchange_code"}:
                score[column] = 1_000_000.0
                future[column] = 2_000_000.0
        out[source] = pd.concat([frame, score, future], ignore_index=True)
    return out


def test_catalogue_has_four_three_signal_families_and_import_only_registration() -> None:
    entries = lattice.cross_sectional_rank_lattice_catalog()

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "cross_sectional_optionality_rank_lattice": 3,
        "cross_sectional_floor_credit_rank_lattice": 3,
        "cross_sectional_liquidity_capacity_rank_lattice": 3,
        "cross_sectional_barrier_geometry_rank_lattice": 3,
    }
    assert set(lattice.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(lattice.KERNEL_NAME) is lattice.FactorMiningCrossSectionalRankLatticeV1
    assert lattice.FactorMiningCrossSectionalRankLatticeV1.__name__ not in factor_operators.__all__
    assert lattice.FactorMiningCrossSectionalRankLatticeV1.requires_stock_panel is False
    assert lattice.FactorMiningCrossSectionalRankLatticeV1.requires_bond_stock_map is False


def test_requirements_are_family_specific_and_explicit() -> None:
    option = lattice.FactorMiningCrossSectionalRankLatticeV1.daily_requirements(
        {"signal": "csl_option_moneyness_stockvol_lattice"}
    )
    barrier = lattice.FactorMiningCrossSectionalRankLatticeV1.daily_requirements(
        {"signal": "csl_barrier_trigger_premium_wedge"}
    )

    assert [item.source for item in option] == ["market_cbond.daily_price", "market_cbond.daily_base"]
    assert "conv_value" in option[1].columns
    assert "cb_call_price" not in option[1].columns
    assert "cb_call_price" in barrier[1].columns
    assert "trigger_price_revise" in barrier[1].columns
    assert all(item.lookback_days >= 8 for item in barrier)


def test_all_signals_build_without_inf_or_constants_on_complete_snapshot() -> None:
    frame = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())

    assert frame.columns.tolist() == [entry.signal for entry in lattice.cross_sectional_rank_lattice_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= len(CODES) - 1).all()
    assert (frame.nunique(dropna=True) > 1).all()


def test_all_signals_ignore_score_day_and_future_daily_mutations() -> None:
    baseline = build_factor_frame(_panel(), _specs(), daily_data=_daily_sources())
    contaminated = build_factor_frame(_panel(), _specs(), daily_data=_contaminate(_daily_sources()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_small_cross_section_missing_fields_duplicates_and_stale_base_fail_closed() -> None:
    small_codes = CODES[: lattice._MIN_CROSS_SECTION - 1]
    small_bare = {_bare(code) for code in small_codes}
    small_sources = {
        source: frame.loc[frame["code"].isin(small_bare)].copy()
        for source, frame in _daily_sources().items()
    }
    small = build_factor_frame(_panel(small_codes), _specs(), daily_data=small_sources)
    assert small.isna().all(axis=None)

    kernel = lattice.FactorMiningCrossSectionalRankLatticeV1()
    sources = _daily_sources()
    missing = sources["market_cbond.daily_base"].drop(columns=["conv_value"])
    with pytest.raises(KeyError, match="conv_value"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": missing},
                params={"signal": "csl_option_moneyness_stockvol_lattice"},
            )
        )

    duplicate = pd.concat(
        [sources["market_cbond.daily_base"], sources["market_cbond.daily_base"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": duplicate},
                params={"signal": "csl_option_moneyness_stockvol_lattice"},
            )
        )

    latest = sources["market_cbond.daily_price"]["trade_date"].max()
    stale = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == _bare(CODES[0]))
            & (sources["market_cbond.daily_base"]["trade_date"] == latest)
        )
    ].copy()
    frame = build_factor_frame(
        _panel(),
        _specs(),
        daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": stale},
    )
    assert frame.loc[(SCORE, CODES[0])].isna().all()
    assert frame.loc[(SCORE, CODES[1])].notna().all()


def test_lattice_formula_uses_centered_cross_sectional_rank_and_unknown_signal_errors() -> None:
    count = lattice._MIN_CROSS_SECTION
    sequence = np.arange(count, dtype="float64")
    snapshot = pd.DataFrame(
        {
            "close_price": np.full(count, 100.0),
            "conv_value": 80.0 + sequence,
            "bond_prem_ratio": 0.04 + 0.001 * sequence,
            "stock_volatility": 0.10 + 0.002 * sequence,
            "duration": 1.0 + 0.1 * sequence,
        }
    )
    observed = lattice._optionality_features(snapshot)

    centered = (sequence + 1.0) / count - 0.5
    assert observed["csl_option_moneyness_stockvol_lattice"].to_numpy() == pytest.approx(centered * centered)
    assert observed["csl_option_premium_duration_lattice"].to_numpy() == pytest.approx(centered * centered)

    with pytest.raises(KeyError, match="unknown signal"):
        lattice.FactorMiningCrossSectionalRankLatticeV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "not_a_rank_lattice_signal"})
        )
