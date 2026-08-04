from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_orderbook_repricing_v1 as repricing
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=9, minutes=37),
        pd.Timedelta(hours=9, minutes=44),
        pd.Timedelta(hours=9, minutes=52),
        pd.Timedelta(hours=10, minutes=1),
        pd.Timedelta(hours=10, minutes=12),
        pd.Timedelta(hours=10, minutes=24),
        pd.Timedelta(hours=10, minutes=38),
        pd.Timedelta(hours=10, minutes=52),
        pd.Timedelta(hours=11, minutes=7),
        pd.Timedelta(hours=11, minutes=22),
        pd.Timedelta(hours=11, minutes=30),
        pd.Timedelta(hours=13),
        pd.Timedelta(hours=13, minutes=9),
        pd.Timedelta(hours=13, minutes=19),
        pd.Timedelta(hours=13, minutes=30),
        pd.Timedelta(hours=13, minutes=42),
        pd.Timedelta(hours=13, minutes=55),
        pd.Timedelta(hours=14, minutes=7),
        pd.Timedelta(hours=14, minutes=18),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _complete_panel(*, code: str = BOND) -> pd.DataFrame:
    bid_ticks = np.asarray([0, 1, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 1, 0, 1, 2, 3, 2])
    ask_ticks = np.asarray([0, 1, 0, -1, 0, 1, 2, 1, 0, 1, 2, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0])
    rows: list[dict[str, object]] = []
    for seq, clock in enumerate(_times()):
        bid1 = 99.92 + 0.01 * float(bid_ticks[seq])
        ask1 = 100.08 + 0.01 * float(ask_ticks[seq])
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": seq,
            "trade_time": DT.normalize() + clock,
        }
        for level in range(1, 6):
            row[f"bid_price{level}"] = bid1 - 0.01 * float(level - 1)
            row[f"ask_price{level}"] = ask1 + 0.01 * float(level - 1)
            row[f"bid_volume{level}"] = float(65 + 7 * level + ((seq * (level + 2)) % 23))
            row[f"ask_volume{level}"] = float(72 + 6 * level + ((seq * (level + 3) + 4) % 19))
        rows.append(row)
    panel = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=repricing.KERNEL_NAME, params={"signal": entry.signal})
        for entry in repricing.factor_mining_orderbook_repricing_v1_catalog()
    ]


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[:2].copy()
    prior["seq"] = prior["seq"] + 100
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)

    lunch = frame.iloc[[0]].copy()
    lunch["seq"] = 1000
    lunch["trade_time"] = DT.normalize() + pd.Timedelta(hours=12)

    after_cutoff = frame.iloc[[-1]].copy()
    after_cutoff["seq"] = 1001
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_cutoff["ask_price1"] = 1.0
    after_cutoff["bid_price1"] = 999.0

    after_1430 = frame.iloc[[-1]].copy()
    after_1430["seq"] = 1002
    after_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    after_1430["ask_price1"] = 1.0
    after_1430["bid_price1"] = 999.0

    contaminated = pd.concat([prior, frame, lunch, after_cutoff, after_1430], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    contaminated.attrs["__build_day__"] = DT.date().isoformat()
    return contaminated


def test_catalogue_has_two_three_signal_reprice_families_and_registry() -> None:
    entries = repricing.factor_mining_orderbook_repricing_v1_catalog()

    assert len(entries) == 6
    assert len({entry.signal for entry in entries}) == 6
    assert Counter(entry.family for entry in entries) == {
        "price_ladder_reprice_direction": 3,
        "reprice_conditioned_depth_relocation": 3,
    }
    assert {entry.kernel for entry in entries} == {repricing.KERNEL_NAME}
    assert set(repricing.FORMULAS) == {entry.signal for entry in entries}
    assert FactorRegistry.get(repricing.KERNEL_NAME) is repricing.FactorMiningOrderbookRepricingV1
    assert repricing.FactorMiningOrderbookRepricingV1.__name__ not in factor_defs.__all__
    assert repricing.FactorMiningOrderbookRepricingV1.daily_requirements() == []


def test_repricing_kernel_builds_dt_code_contract_without_inf_or_counter_input() -> None:
    frame = build_factor_frame(_complete_panel(), _specs(), workers=2)

    assert frame.columns.tolist() == [entry.signal for entry in repricing.factor_mining_orderbook_repricing_v1_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert int(frame.notna().sum(axis=1).iloc[0]) == 6


def test_repricing_ignores_relabelled_prior_lunch_and_rows_after_1429() -> None:
    baseline = build_factor_frame(_complete_panel(), _specs(), workers=2)
    contaminated = build_factor_frame(_with_out_of_contract_rows(_complete_panel()), _specs(), workers=2)

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_ladder_event_at_t_never_uses_a_later_quote() -> None:
    frame = repricing._score_day_frame(FactorComputeContext(panel=_complete_panel()))
    book = repricing._book(frame)
    assert book is not None
    original = repricing._ladder_directions(book.bid_price, book.sessions)
    assert original is not None

    extended_prices = np.vstack([book.bid_price, book.bid_price[-1] * 1.005])
    extended_sessions = np.r_[book.sessions, book.sessions[-1]]
    extended = repricing._ladder_directions(extended_prices, extended_sessions)

    assert extended is not None
    np.testing.assert_array_equal(extended[:-1], original)


def test_crossed_book_fails_closed_for_both_families() -> None:
    panel = _complete_panel().copy()
    panel.iloc[7, panel.columns.get_loc("ask_price1")] = 1.0
    panel.iloc[7, panel.columns.get_loc("bid_price1")] = 999.0
    panel.attrs["__build_day__"] = DT.date().isoformat()

    frame = build_factor_frame(panel, _specs(), workers=2)

    assert frame.index.tolist() == [(DT, BOND)]
    assert frame.isna().all(axis=None)


def test_zero_depth_denominators_fail_closed_only_for_depth_relocation_family() -> None:
    panel = _complete_panel().copy()
    depth_columns = [column for column in panel.columns if "volume" in column]
    panel.loc[:, depth_columns] = 0.0
    panel.attrs["__build_day__"] = DT.date().isoformat()
    entries = repricing.factor_mining_orderbook_repricing_v1_catalog()
    direction_specs = [spec for spec, entry in zip(_specs(), entries, strict=True) if entry.family == "price_ladder_reprice_direction"]
    depth_specs = [spec for spec, entry in zip(_specs(), entries, strict=True) if entry.family == "reprice_conditioned_depth_relocation"]

    direction = build_factor_frame(panel, direction_specs)
    depth = build_factor_frame(panel, depth_specs)

    assert direction.notna().all(axis=None)
    assert depth.isna().all(axis=None)


def test_missing_ladder_column_too_few_events_and_unknown_signal_are_explicit() -> None:
    missing = _complete_panel().drop(columns=["ask_price4"])
    missing.attrs["__build_day__"] = DT.date().isoformat()
    with pytest.raises(KeyError, match="ask_price4"):
        repricing.FactorMiningOrderbookRepricingV1().compute(
            FactorComputeContext(panel=missing, params={"signal": "lrd_bid_reprice_direction_persistence"})
        )

    no_reprice = _complete_panel().copy()
    for level in range(1, 6):
        no_reprice.loc[:, f"bid_price{level}"] = 99.92 - 0.01 * float(level - 1)
        no_reprice.loc[:, f"ask_price{level}"] = 100.08 + 0.01 * float(level - 1)
    no_reprice.attrs["__build_day__"] = DT.date().isoformat()
    assert build_factor_frame(no_reprice, _specs()).isna().all(axis=None)

    with pytest.raises(KeyError, match="unknown signal"):
        repricing.FactorMiningOrderbookRepricingV1().compute(
            FactorComputeContext(panel=_complete_panel(), params={"signal": "not_a_reprice_signal"})
        )
