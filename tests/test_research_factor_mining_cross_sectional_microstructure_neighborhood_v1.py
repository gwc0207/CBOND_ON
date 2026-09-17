from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators as factor_operators
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_sectional_microstructure_neighborhood_v1 as neighborhood,
)
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"{110000 + index:06d}.SH" for index in range(48))


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=9, minutes=35),
        pd.Timedelta(hours=9, minutes=42),
        pd.Timedelta(hours=9, minutes=50),
        pd.Timedelta(hours=10),
        pd.Timedelta(hours=10, minutes=12),
        pd.Timedelta(hours=10, minutes=25),
        pd.Timedelta(hours=10, minutes=38),
        pd.Timedelta(hours=10, minutes=52),
        pd.Timedelta(hours=11, minutes=8),
        pd.Timedelta(hours=11, minutes=22),
        pd.Timedelta(hours=11, minutes=30),
        pd.Timedelta(hours=13),
        pd.Timedelta(hours=13, minutes=9),
        pd.Timedelta(hours=13, minutes=18),
        pd.Timedelta(hours=13, minutes=28),
        pd.Timedelta(hours=13, minutes=39),
        pd.Timedelta(hours=13, minutes=50),
        pd.Timedelta(hours=14),
        pd.Timedelta(hours=14, minutes=8),
        pd.Timedelta(hours=14, minutes=16),
        pd.Timedelta(hours=14, minutes=23),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _panel(codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clocks = _times()
    for code_index, code in enumerate(codes):
        upper = 105.0 + 0.03 * (code_index % 5)
        lower = 95.0 - 0.03 * (code_index % 5)
        base_mid = 100.0 + 0.06 * code_index
        trade_total = 0.0
        prior_ask = base_mid + 0.04
        prior_bid = base_mid - 0.04
        for sequence, clock in enumerate(clocks):
            mid = base_mid + 0.018 * sequence + 0.09 * np.sin(0.51 * sequence + 0.37 * code_index)
            spread = 0.06 + 0.006 * (code_index % 5) + 0.004 * (sequence % 3)
            # Some, but not all, physical clocks retain the prior quote.  This
            # gives the neighbourhood anchors genuine cross-sectional
            # variation without weakening their strict visible-clock contract.
            # The modulo period differs by code, so quote-update rates do not
            # collapse to one constant value across the synthetic universe.
            quote_retained = sequence > 0 and (sequence + 2 * code_index) % (4 + code_index % 3) == 0
            if quote_retained:
                ask1 = prior_ask
                bid1 = prior_bid
                mid = (ask1 + bid1) / 2.0
            else:
                ask1 = float(mid + spread / 2.0)
                bid1 = float(mid - spread / 2.0)
            if sequence == 0:
                last = float(mid)
            elif (sequence + code_index) % 4 == 0:
                last = prior_ask
            elif (sequence + 2 * code_index) % 5 == 0:
                last = prior_bid
            else:
                last = float(mid)
            if sequence == len(clocks) - 1 and code_index % 11 == 0:
                last = upper
                bid1 = upper
                ask1 = upper + 0.035
            elif sequence == len(clocks) - 1 and code_index % 13 == 0:
                last = lower
                ask1 = lower
                bid1 = lower - 0.035
            trade_step = float(1 + ((3 * sequence + 2 * code_index) % 8))
            if (sequence + code_index) % 7 == 0:
                trade_step = 0.0
            trade_total += trade_step
            row: dict[str, object] = {
                "dt": DT,
                "code": code,
                "seq": sequence,
                "trade_time": DT.normalize() + clock,
                "last": float(last),
                "high_limited": upper,
                "low_limited": lower,
                "num_trades": trade_total,
                "ask_price1": ask1,
                "bid_price1": bid1,
                "ask_volume1": float(25 + ((3 * sequence + 5 * code_index) % 29)),
                "bid_volume1": float(28 + ((5 * sequence + 3 * code_index) % 31)),
            }
            for level in range(2, 6):
                row[f"ask_price{level}"] = ask1 + 0.011 * level
                row[f"bid_price{level}"] = bid1 - 0.010 * level
                row[f"ask_volume{level}"] = float(
                    20 + 4 * level + ((sequence * (level + 1) + code_index) % 23)
                )
                row[f"bid_volume{level}"] = float(
                    22 + 5 * level + ((2 * sequence * level + 3 * code_index) % 19)
                )
            rows.append(row)
            prior_ask = ask1
            prior_bid = bid1
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=neighborhood.KERNEL_NAME, params={"signal": entry.signal})
        for entry in neighborhood.factor_mining_cross_sectional_microstructure_neighborhood_v1_catalog()
    ]


def _out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    source = frame.loc[frame["code"] == CODES[0]].iloc[[5]].copy()
    prior = source.copy()
    prior["seq"] = 10_001
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 1.0
    prior["num_trades"] = 1_000_000.0

    lunch = source.copy()
    lunch["seq"] = 10_002
    lunch["trade_time"] = DT.normalize() + pd.Timedelta(hours=12)
    lunch["last"] = 1.0

    after_1429 = source.copy()
    after_1429["seq"] = 10_003
    after_1429["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_1429["last"] = 1.0

    cutoff = source.copy()
    cutoff["seq"] = 10_004
    cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    cutoff["last"] = 1.0

    out = pd.concat([frame, prior, lunch, after_1429, cutoff], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _counter_reset(panel: pd.DataFrame, code: str) -> pd.DataFrame:
    frame = panel.reset_index().copy()
    current = frame.index[(frame["code"] == code) & (frame["seq"] == 12)]
    previous = frame.index[(frame["code"] == code) & (frame["seq"] == 11)]
    assert len(current) == 1 and len(previous) == 1
    frame.loc[current[0], "num_trades"] = float(frame.loc[previous[0], "num_trades"]) - 1.0
    out = frame.set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_catalogue_has_four_three_signal_families_and_import_only_registry() -> None:
    entries = neighborhood.factor_mining_cross_sectional_microstructure_neighborhood_v1_catalog()

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "csn_limit_queue_local_dislocation": 3,
        "csn_passive_queue_local_dislocation": 3,
        "csn_quote_initiation_local_dislocation": 3,
        "csn_depth_cascade_local_dislocation": 3,
    }
    assert {entry.kernel for entry in entries} == {neighborhood.KERNEL_NAME}
    assert FactorRegistry.get(neighborhood.KERNEL_NAME) is neighborhood.FactorMiningCrossSectionalMicrostructureNeighborhoodV1
    assert neighborhood.FactorMiningCrossSectionalMicrostructureNeighborhoodV1.__name__ not in factor_operators.__all__
    assert neighborhood.FactorMiningCrossSectionalMicrostructureNeighborhoodV1.requires_stock_panel is False
    assert neighborhood.FactorMiningCrossSectionalMicrostructureNeighborhoodV1.requires_bond_stock_map is False
    assert neighborhood.FactorMiningCrossSectionalMicrostructureNeighborhoodV1.daily_requirements() == []
    assert set(neighborhood.FORMULAS) == {entry.signal for entry in entries}


def test_neighborhood_kernel_builds_all_signals_without_inf_or_constants() -> None:
    frame = build_factor_frame(_panel(), _specs())

    assert frame.columns.tolist() == [
        entry.signal for entry in neighborhood.factor_mining_cross_sectional_microstructure_neighborhood_v1_catalog()
    ]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert (frame.notna().sum(axis=0) >= len(CODES) - 2).all()
    assert (frame.nunique(dropna=True) > 1).all()


def test_neighborhood_ignores_relabelled_prior_lunch_and_rows_after_1429() -> None:
    baseline = build_factor_frame(_panel(), _specs())
    contaminated = build_factor_frame(_out_of_contract_rows(_panel()), _specs())

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_duplicate_physical_clock_fails_closed_for_one_code_without_cross_section_fallback() -> None:
    frame = _panel().reset_index()
    duplicate = frame.loc[frame["code"] == CODES[0]].iloc[[8]].copy()
    duplicate["seq"] = 20_001
    broken = pd.concat([frame, duplicate], ignore_index=True).set_index(["dt", "code", "seq"])
    broken.attrs["__build_day__"] = DT.date().isoformat()

    observed = build_factor_frame(broken, _specs())

    assert observed.loc[(DT, CODES[0])].isna().all()
    assert observed.loc[(DT, CODES[1])].notna().any()
    assert not np.isinf(observed.to_numpy(dtype="float64")).any()


def test_trade_counter_reset_only_suppresses_trade_dependent_neighborhood_families() -> None:
    observed = build_factor_frame(_counter_reset(_panel(), CODES[0]), _specs())
    entries = neighborhood.factor_mining_cross_sectional_microstructure_neighborhood_v1_catalog()
    trade_dependent = [
        entry.signal
        for entry in entries
        if entry.family
        in {
            "csn_limit_queue_local_dislocation",
            "csn_passive_queue_local_dislocation",
            "csn_quote_initiation_local_dislocation",
        }
    ]
    depth_only = [entry.signal for entry in entries if entry.family == "csn_depth_cascade_local_dislocation"]

    assert observed.loc[(DT, CODES[0]), trade_dependent].isna().all()
    assert observed.loc[(DT, CODES[0]), depth_only].notna().all()
    assert observed.loc[(DT, CODES[1])].notna().all()


def test_neighborhood_requires_explicit_fields_without_fallback() -> None:
    panel = _panel().drop(columns=["num_trades"])
    panel.attrs["__build_day__"] = DT.date().isoformat()

    with pytest.raises(KeyError, match="num_trades"):
        neighborhood.FactorMiningCrossSectionalMicrostructureNeighborhoodV1().compute(
            FactorComputeContext(panel=panel, params={"signal": "csn_pql_retention_neighbor_gap"})
        )


def test_neighborhood_requires_a_full_finite_cross_section() -> None:
    observed = build_factor_frame(_panel(CODES[: neighborhood._MIN_CROSS_SECTION - 1]), _specs())

    assert observed.isna().all(axis=None)


def test_unknown_signal_is_explicit_error() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        neighborhood.FactorMiningCrossSectionalMicrostructureNeighborhoodV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "not_a_neighborhood_signal"})
        )
