from __future__ import annotations

from collections import Counter
from datetime import time as dt_time

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_intraday_conversion_parity_dynamics_v1 as parity
from cbond_on.domain.factors.spec import FactorSpec


SCORE = pd.Timestamp("2026-07-30 14:30:00")
BONDS = tuple(f"110{index:03d}.SH" for index in range(16))
STOCKS = tuple(f"600{index:03d}.SH" for index in range(16))


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _exchange(code: str) -> str:
    return code.rsplit(".", 1)[1]


def _times() -> list[pd.Timestamp]:
    day = SCORE.normalize()
    morning = list(pd.date_range(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=11, minutes=30), freq="5min"))
    afternoon = list(pd.date_range(day + pd.Timedelta(hours=13), day + pd.Timedelta(hours=14, minutes=25), freq="5min"))
    return [*morning, *afternoon, day + pd.Timedelta(hours=14, minutes=29)]


def _panel(codes: tuple[str, ...], *, is_stock: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clocks = _times()
    for code_index, code in enumerate(codes):
        conv_price = 10.0 + 0.15 * code_index
        stock_base = conv_price * (1.12 + 0.006 * (code_index % 5))
        for sequence, timestamp in enumerate(clocks):
            position = float(sequence)
            stock = stock_base * np.exp(
                0.0001 * position
                + 0.0045 * np.sin(position / 2.7 + 0.25 * code_index)
                + 0.0013 * np.cos(position / 4.1 + 0.18 * code_index)
            )
            wedge = (
                -0.055
                + 0.006 * code_index
                + 0.024 * np.sin(position / 3.1 + 0.37 * code_index)
                + 0.011 * np.cos(position / 1.9 + 0.11 * code_index)
                + 0.0007 * position
            )
            last = stock if is_stock else (100.0 * stock / conv_price) * np.exp(wedge)
            rows.append(
                {
                    "dt": SCORE,
                    "code": code,
                    "seq": sequence,
                    "trade_time": timestamp,
                    "last": last,
                }
            )
    return pd.DataFrame(rows).set_index(["dt", "code", "seq"])


def _mapping(*, future: bool = False, mismatch_code: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bond, stock in zip(BONDS, STOCKS, strict=True):
        mapped = STOCKS[-1] if bond == mismatch_code else stock
        row: dict[str, object] = {
            "code": bond,
            "stock_code": mapped,
            "trade_date": SCORE.normalize() + pd.offsets.BDay(1) if future else SCORE.normalize(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _daily_sources() -> dict[str, pd.DataFrame]:
    prior = SCORE.normalize() - pd.offsets.BDay(1)
    older = prior - pd.offsets.BDay(1)
    future = SCORE.normalize() + pd.offsets.BDay(1)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, (bond, stock) in enumerate(zip(BONDS, STOCKS, strict=True)):
        conv_price = 10.0 + 0.15 * code_index
        for day, close in ((older, 112.0 + code_index), (prior, 113.0 + code_index)):
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(bond),
                    "exchange_code": _exchange(bond),
                    "close_price": close,
                }
            )
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(bond),
                    "exchange_code": _exchange(bond),
                    "cb_conv_price": conv_price,
                    "stock_code": stock,
                }
            )
        price_rows.append(
            {
                "trade_date": SCORE.normalize(),
                "code": _bare(bond),
                "exchange_code": _exchange(bond),
                "close_price": 1_000_000.0,
            }
        )
        base_rows.extend(
            [
                {
                    "trade_date": SCORE.normalize(),
                    "code": _bare(bond),
                    "exchange_code": _exchange(bond),
                    "cb_conv_price": 1_000_000.0,
                    "stock_code": STOCKS[-1],
                },
                {
                    "trade_date": future,
                    "code": _bare(bond),
                    "exchange_code": _exchange(bond),
                    "cb_conv_price": 2_000_000.0,
                    "stock_code": STOCKS[-1],
                },
            ]
        )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=parity.KERNEL_NAME, params={"signal": entry.signal})
        for entry in parity.intraday_conversion_parity_dynamics_catalog()
    ]


def _build(
    *,
    bond_panel: pd.DataFrame | None = None,
    stock_panel: pd.DataFrame | None = None,
    mapping: pd.DataFrame | None = None,
    daily_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    return build_factor_frame(
        _panel(BONDS, is_stock=False) if bond_panel is None else bond_panel,
        _specs(),
        stock_panel=_panel(STOCKS, is_stock=True) if stock_panel is None else stock_panel,
        bond_stock_map=_mapping() if mapping is None else mapping,
        daily_data=_daily_sources() if daily_data is None else daily_data,
    )


def _with_post_cutoff_ticks(panel: pd.DataFrame, *, multiplier: float) -> pd.DataFrame:
    extra = panel.reset_index().groupby(["dt", "code"], sort=False).tail(1).copy()
    extra["seq"] = extra["seq"].astype(int) + 10_000
    extra["trade_time"] = SCORE.normalize() + pd.Timedelta(hours=14, minutes=30)
    extra["last"] = pd.to_numeric(extra["last"], errors="coerce") * multiplier
    return pd.concat([panel.reset_index(), extra], ignore_index=True).set_index(["dt", "code", "seq"])


def test_catalogue_has_three_nonstatic_families_and_research_only_registration() -> None:
    entries = parity.intraday_conversion_parity_dynamics_catalog()

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "intraday_parity_wedge_trajectory": 3,
        "intraday_parity_wedge_timing": 3,
        "stock_conditioned_parity_adjustment": 3,
    }
    assert set(parity.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(parity.KERNEL_NAME) is parity.FactorMiningIntradayConversionParityDynamicsV1
    assert parity.FactorMiningIntradayConversionParityDynamicsV1.__name__ not in factor_defs.__all__
    assert parity.FactorMiningIntradayConversionParityDynamicsV1.requires_stock_panel is True
    assert parity.FactorMiningIntradayConversionParityDynamicsV1.requires_bond_stock_map is True


def test_requirements_use_tminus1_contract_and_independent_price_anchor() -> None:
    requirements = parity.FactorMiningIntradayConversionParityDynamicsV1.daily_requirements()

    assert [item.source for item in requirements] == ["market_cbond.daily_price", "market_cbond.daily_base"]
    assert requirements[0].columns == ("exchange_code", "close_price")
    assert set(requirements[1].columns) == {"exchange_code", "cb_conv_price", "stock_code"}
    assert all(item.lookback_days >= 2 for item in requirements)


def test_all_signals_build_without_inf_or_constants_on_complete_context() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in parity.intraday_conversion_parity_dynamics_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in BONDS]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= len(BONDS) - 1).all()
    assert (frame.nunique(dropna=True) > 1).all()


def test_post_cutoff_and_score_day_or_future_daily_mutations_are_ignored() -> None:
    baseline = _build()
    contaminated = _build(
        bond_panel=_with_post_cutoff_ticks(_panel(BONDS, is_stock=False), multiplier=50.0),
        stock_panel=_with_post_cutoff_ticks(_panel(STOCKS, is_stock=True), multiplier=0.02),
        mapping=pd.concat([_mapping(), _mapping(future=True)], ignore_index=True),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_missing_tminus1_contract_or_map_mismatch_fails_closed_per_code() -> None:
    sources = _daily_sources()
    prior = SCORE.normalize() - pd.offsets.BDay(1)
    no_contract = sources["market_cbond.daily_base"].loc[
        ~(
            (sources["market_cbond.daily_base"]["code"] == _bare(BONDS[0]))
            & (sources["market_cbond.daily_base"]["trade_date"] == prior)
        )
    ].copy()
    frame = _build(daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": no_contract})
    assert frame.loc[(SCORE, BONDS[0])].isna().all()
    assert frame.loc[(SCORE, BONDS[1])].notna().all()

    mismatch = _build(mapping=_mapping(mismatch_code=BONDS[0]))
    assert mismatch.loc[(SCORE, BONDS[0])].isna().all()
    assert mismatch.loc[(SCORE, BONDS[1])].notna().all()


def test_missing_daily_field_duplicate_anchor_and_future_only_map_are_explicit() -> None:
    kernel = parity.FactorMiningIntradayConversionParityDynamicsV1()
    sources = _daily_sources()
    missing = sources["market_cbond.daily_base"].drop(columns=["cb_conv_price"])
    with pytest.raises(KeyError, match="cb_conv_price"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(BONDS, is_stock=False),
                stock_panel=_panel(STOCKS, is_stock=True),
                bond_stock_map=_mapping(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": missing},
                params={"signal": "icpd_wedge_segment_convergence"},
            )
        )

    duplicate_price = pd.concat(
        [sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(BONDS, is_stock=False),
                stock_panel=_panel(STOCKS, is_stock=True),
                bond_stock_map=_mapping(),
                daily_data={"market_cbond.daily_price": duplicate_price, "market_cbond.daily_base": sources["market_cbond.daily_base"]},
                params={"signal": "icpd_wedge_segment_convergence"},
            )
        )

    future_only = _build(mapping=_mapping(future=True))
    assert future_only.isna().all().all()


def test_gap_does_not_create_synthetic_returns_or_trajectory_segments() -> None:
    stock = _panel(STOCKS, is_stock=True).reset_index()
    keep = stock["trade_time"].dt.minute % 10 == 0
    gapped_stock = stock.loc[keep].set_index(["dt", "code", "seq"])
    trajectory = _build(stock_panel=gapped_stock)[list(parity._TRAJECTORY_SIGNALS)]
    conditioned = _build(stock_panel=gapped_stock)[list(parity._STOCK_CONDITIONED_SIGNALS)]

    assert trajectory.isna().all().all()
    assert conditioned.isna().all().all()


def test_contract_parity_path_uses_exact_wedge_and_never_crosses_lunch() -> None:
    day = SCORE.normalize()
    times = [
        day + pd.Timedelta(hours=9, minutes=30 + 5 * index)
        for index in range(6)
    ] + [
        day + pd.Timedelta(hours=13, minutes=5 * index)
        for index in range(6)
    ]
    wedges = np.asarray([0.10, 0.08, 0.05, 0.02, -0.01, -0.03, -0.02, 0.00, 0.03, 0.01, -0.02, -0.04])
    stock = 12.0
    bond_frame = pd.DataFrame(
        {
            "dt": SCORE,
            "code": BONDS[0],
            "seq": range(len(times)),
            "trade_time": times,
            "last": 120.0 * np.exp(wedges),
        }
    ).set_index(["dt", "code", "seq"])
    stock_frame = pd.DataFrame(
        {
            "dt": SCORE,
            "code": STOCKS[0],
            "seq": range(len(times)),
            "trade_time": times,
            "last": stock,
        }
    ).set_index(["dt", "code", "seq"])
    path = parity._parity_path(
        parity._strict_physical_frame(bond_frame, score_date=SCORE.normalize(), owner="panel"),
        parity._strict_physical_frame(stock_frame, score_date=SCORE.normalize(), owner="stock_panel"),
        conv_price=10.0,
    )

    assert path["wedge"].to_numpy() == pytest.approx(wedges)
    assert path.loc[day + pd.Timedelta(hours=13), "__contiguous"] == np.False_
    metrics = parity._trajectory_metrics(path)
    expected_convergence = np.mean([abs(wedges[0]) - abs(wedges[5]), abs(wedges[6]) - abs(wedges[-1])])
    assert metrics["icpd_wedge_segment_convergence"] == pytest.approx(expected_convergence)
    assert (path.index.to_series().diff().dropna() == pd.Timedelta(hours=3, minutes=5)).any()
    assert parity._continuous_coordinate(day + pd.Timedelta(hours=9, minutes=30)) == pytest.approx(0.0)
    assert parity._continuous_coordinate(day + pd.Timedelta(hours=14, minutes=25)) < 1.0
    assert (day + pd.Timedelta(hours=14, minutes=30)).time() > dt_time(14, 29)
