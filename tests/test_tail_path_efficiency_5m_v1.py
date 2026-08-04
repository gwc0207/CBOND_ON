from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs.tail_path_efficiency_5m_v1 import TailPathEfficiency5mV1Factor


DT = pd.Timestamp("2026-07-30 14:30:00")
CODE = "110001.SH"


def _rows(
    mids: list[float],
    *,
    code: str = CODE,
    day: str = "2026-07-30",
) -> list[dict[str, object]]:
    return [
        {
            "dt": DT,
            "code": code,
            "seq": seq,
            "trade_time": pd.Timestamp(f"{day} 14:{24 + seq:02d}:00"),
            "bid_price1": mid - 0.01,
            "ask_price1": mid + 0.01,
        }
        for seq, mid in enumerate(mids)
    ]


def _compute(rows: list[dict[str, object]]) -> pd.Series:
    frame = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    return TailPathEfficiency5mV1Factor().compute(FactorComputeContext(panel=frame))


def test_tail_path_efficiency_is_registered_and_panel_only() -> None:
    assert FactorRegistry.get("tail_path_efficiency_5m_v1") is TailPathEfficiency5mV1Factor
    assert TailPathEfficiency5mV1Factor.requires_stock_panel is False
    assert TailPathEfficiency5mV1Factor.requires_bond_stock_map is False
    assert TailPathEfficiency5mV1Factor.daily_requirements() == []


def test_tail_path_efficiency_is_one_for_a_strictly_monotone_mid_path() -> None:
    actual = _compute(_rows([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]))
    assert actual.name == "tail_path_efficiency_5m_v1"
    assert actual.loc[(DT, CODE)] == pytest.approx(1.0)


def test_tail_path_efficiency_uses_signed_net_move_over_total_log_path() -> None:
    mids = [100.0, 102.0, 101.0, 103.0, 102.0, 104.0]
    actual = _compute(_rows(mids))
    log_mid = np.log(np.asarray(mids, dtype=float))
    expected = (log_mid[-1] - log_mid[0]) / np.abs(np.diff(log_mid)).sum()
    assert actual.loc[(DT, CODE)] == pytest.approx(expected)


def test_tail_path_efficiency_excludes_post_1430_and_cross_day_rows() -> None:
    rows = _rows([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    rows.extend(
        [
            {
                "dt": DT,
                "code": CODE,
                "seq": 99,
                "trade_time": pd.Timestamp("2026-07-30 14:30:00"),
                "bid_price1": 1.0,
                "ask_price1": 999.0,
            },
            {
                "dt": DT,
                "code": CODE,
                "seq": 100,
                "trade_time": pd.Timestamp("2026-07-29 14:29:00"),
                "bid_price1": 1.0,
                "ask_price1": 999.0,
            },
        ]
    )
    actual = _compute(rows)
    assert actual.loc[(DT, CODE)] == pytest.approx(1.0)


def test_tail_path_efficiency_requires_every_clock_minute_and_nonzero_path() -> None:
    missing = _rows([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    del missing[2]
    static = _rows([100.0] * 6, code="110002.SH")
    actual = _compute(missing + static)
    assert math.isnan(actual.loc[(DT, CODE)])
    assert math.isnan(actual.loc[(DT, "110002.SH")])
