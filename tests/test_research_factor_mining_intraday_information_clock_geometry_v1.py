from __future__ import annotations

from collections import Counter
import warnings

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import defs as factor_defs
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import (
    research_factor_mining_intraday_information_clock_geometry_v1 as geometry,
)
from cbond_on.domain.factors.spec import FactorSpec


DT = pd.Timestamp("2026-07-30 14:30:00")
CODES = ("110001.SH", "110002.SZ", "113001.SH")


def _times() -> list[pd.Timedelta]:
    return [
        pd.Timedelta(hours=9, minutes=30),
        pd.Timedelta(hours=9, minutes=37),
        pd.Timedelta(hours=9, minutes=46),
        pd.Timedelta(hours=9, minutes=57),
        pd.Timedelta(hours=10, minutes=8),
        pd.Timedelta(hours=10, minutes=19),
        pd.Timedelta(hours=10, minutes=31),
        pd.Timedelta(hours=10, minutes=44),
        pd.Timedelta(hours=10, minutes=58),
        pd.Timedelta(hours=11, minutes=12),
        pd.Timedelta(hours=11, minutes=27),
        pd.Timedelta(hours=11, minutes=30),
        pd.Timedelta(hours=13, minutes=3),
        pd.Timedelta(hours=13, minutes=12),
        pd.Timedelta(hours=13, minutes=24),
        pd.Timedelta(hours=13, minutes=37),
        pd.Timedelta(hours=13, minutes=51),
        pd.Timedelta(hours=14, minutes=4),
        pd.Timedelta(hours=14, minutes=18),
        pd.Timedelta(hours=14, minutes=29),
    ]


def _complete_panel(*, code: str, style: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cumulative_trades = 0.0
    base = 100.0 + 3.0 * float(style)
    for sequence, clock in enumerate(_times()):
        phase = 0.0017 * np.sin(0.67 * sequence + 0.73 * style) + 0.00042 * sequence
        mid_wobble = 0.00075 * np.cos(1.11 * sequence + 0.39 * style)
        last = base * (1.0 + phase + 0.00031 * np.sin(1.43 * sequence + style))
        mid = base * (1.0 + phase + mid_wobble)
        spread = base * (0.00055 + 0.00004 * ((3 * sequence + style) % 5))
        bid1 = mid - spread / 2.0
        ask1 = mid + spread / 2.0
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": sequence,
            "trade_time": DT.normalize() + clock,
            "last": float(last),
        }
        for level in range(1, 6):
            row[f"bid_price{level}"] = bid1 - 0.01 * float(level - 1)
            row[f"ask_price{level}"] = ask1 + 0.01 * float(level - 1)
            row[f"bid_volume{level}"] = float(
                40 + 5 * level + ((sequence * (level + 2) + 3 * style) % 29)
            )
            row[f"ask_volume{level}"] = float(
                47 + 4 * level + ((sequence * (2 * level + 1) + 5 * style) % 31)
            )
        cumulative_trades += float(2 + ((sequence * (style + 2) + style) % 9))
        row["num_trades"] = cumulative_trades
        rows.append(row)
    out = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _panel() -> pd.DataFrame:
    out = pd.concat(
        [
            _complete_panel(code=code, style=index + 1).reset_index()
            for index, code in enumerate(CODES)
        ],
        ignore_index=True,
    ).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name=entry.signal,
            factor=geometry.KERNEL_NAME,
            params={"signal": entry.signal},
        )
        for entry in geometry.factor_mining_catalog()
    ]


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.groupby("code", sort=False).head(1).copy()
    prior["seq"] = prior["seq"] + 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 9_999.0
    prior["num_trades"] = -1.0

    lunch = frame.groupby("code", sort=False).head(1).copy()
    lunch["seq"] = lunch["seq"] + 2_000
    lunch["trade_time"] = DT.normalize() + pd.Timedelta(hours=12)
    lunch["last"] = 8_888.0

    after_cutoff = frame.groupby("code", sort=False).tail(1).copy()
    after_cutoff["seq"] = after_cutoff["seq"] + 3_000
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(
        hours=14, minutes=29, seconds=30
    )
    after_cutoff["last"] = 7_777.0
    after_cutoff["ask_price1"] = 1.0
    after_cutoff["bid_price1"] = 999.0
    after_cutoff["num_trades"] = -2.0

    after_1430 = frame.groupby("code", sort=False).tail(1).copy()
    after_1430["seq"] = after_1430["seq"] + 4_000
    after_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    after_1430["last"] = 6_666.0
    after_1430["ask_price1"] = 1.0
    after_1430["bid_price1"] = 999.0

    out = pd.concat(
        [frame, prior, lunch, after_cutoff, after_1430], ignore_index=True
    ).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_catalogue_has_four_three_signal_families_and_registered_kernel() -> None:
    entries = geometry.factor_mining_catalog()

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "information_clock_curve_trajectory": 3,
        "information_clock_channel_simplex": 3,
        "price_quote_mass_transport": 3,
        "book_discovery_manifold": 3,
    }
    assert {entry.kernel for entry in entries} == {geometry.KERNEL_NAME}
    assert set(geometry.FORMULAS) == {entry.signal for entry in entries}
    assert (
        FactorRegistry.get(geometry.KERNEL_NAME)
        is geometry.FactorMiningIntradayInformationClockGeometryV1
    )
    assert (
        geometry.FactorMiningIntradayInformationClockGeometryV1.__name__
        not in factor_defs.__all__
    )
    assert (
        geometry.FactorMiningIntradayInformationClockGeometryV1.daily_requirements()
        == []
    )


def test_geometry_kernel_builds_dt_code_contract_without_inf_or_cross_sectional_constants() -> (
    None
):
    frame = build_factor_frame(_panel(), _specs(), workers=2)

    assert frame.columns.tolist() == [
        entry.signal for entry in geometry.factor_mining_catalog()
    ]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, code) for code in CODES]
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame.notna().all(axis=None)
    assert (frame.nunique(dropna=True) > 1).all()


def test_geometry_ignores_relabelled_prior_lunch_and_after_1429_rows() -> None:
    baseline = build_factor_frame(_panel(), _specs(), workers=2)
    contaminated = build_factor_frame(
        _with_out_of_contract_rows(_panel()), _specs(), workers=2
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_counter_reset_fails_closed_only_for_counter_based_families() -> None:
    panel = _panel().copy()
    first_code_index = panel.index.get_level_values("code") == CODES[0]
    locations = np.flatnonzero(first_code_index)
    panel.iloc[locations[9], panel.columns.get_loc("num_trades")] = 0.0
    panel.attrs["__build_day__"] = DT.date().isoformat()

    frame = build_factor_frame(panel, _specs(), workers=2)
    entries = geometry.factor_mining_catalog()
    counter_signals = [
        entry.signal
        for entry in entries
        if entry.family
        in {"information_clock_curve_trajectory", "information_clock_channel_simplex"}
    ]
    quote_book_signals = [
        entry.signal for entry in entries if entry.signal not in counter_signals
    ]

    assert frame.loc[(DT, CODES[0]), counter_signals].isna().all()
    assert frame.loc[(DT, CODES[0]), quote_book_signals].notna().all()
    assert frame.loc[(DT, CODES[1]), :].notna().all()


def test_zero_information_clock_step_fails_closed_without_runtime_warning() -> None:
    panel = _complete_panel(code=CODES[0], style=1).copy()
    previous, current = 5, 6
    fields = [
        "last",
        "ask_price1",
        "bid_price1",
        "ask_volume1",
        "bid_volume1",
        "num_trades",
    ]
    for field in fields:
        panel.iloc[current, panel.columns.get_loc(field)] = panel.iloc[
            previous, panel.columns.get_loc(field)
        ]
    panel.attrs["__build_day__"] = DT.date().isoformat()
    curve_specs = [
        spec
        for spec, entry in zip(_specs(), geometry.factor_mining_catalog(), strict=True)
        if entry.family == "information_clock_curve_trajectory"
    ]

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", RuntimeWarning)
        frame = build_factor_frame(panel, curve_specs)

    assert frame.isna().all(axis=None)
    assert not [
        warning for warning in observed if issubclass(warning.category, RuntimeWarning)
    ]


def test_missing_l5_book_field_is_local_to_book_manifold_and_unknown_signal_fails_fast() -> (
    None
):
    panel = _panel().drop(columns=["ask_price4"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    transport_specs = [
        spec
        for spec, entry in zip(_specs(), geometry.factor_mining_catalog(), strict=True)
        if entry.family == "price_quote_mass_transport"
    ]

    transport = build_factor_frame(panel, transport_specs)
    assert transport.notna().all(axis=None)
    with pytest.raises(KeyError, match="ask_price4"):
        geometry.FactorMiningIntradayInformationClockGeometryV1().compute(
            FactorComputeContext(
                panel=panel, params={"signal": "icg_book_discovery_loop_area"}
            )
        )
    with pytest.raises(KeyError, match="unknown signal"):
        geometry.FactorMiningIntradayInformationClockGeometryV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "icg_unknown"})
        )
