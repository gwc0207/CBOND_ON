from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.defs import research_factor_mining_underlying_cohort_intraday_rank_state_v1 as rank_state
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
STOCKS = tuple(f"{600001 + index:06d}.SH" for index in range(36))
PRIMARY_STOCK = STOCKS[4]
OTHER_STOCK = STOCKS[-5]


def _times() -> list[pd.Timedelta]:
    morning = [pd.Timedelta(hours=9, minutes=30 + 5 * index) for index in range(24)]
    afternoon = [pd.Timedelta(hours=13, minutes=5 * index) for index in range(18)]
    return morning + afternoon


def _stock_path(*, code: str, ordinal: int) -> pd.DataFrame:
    price = 10.0 + 0.05 * ordinal
    trades = 0.0
    amount = 0.0
    rows: list[dict[str, object]] = []
    for seq, clock in enumerate(_times()):
        early = seq <= 12
        late = seq >= 30
        early_slope = 0.00008 * ((ordinal % 7) - 3)
        late_slope = 0.00009 * (((3 * ordinal) % 9) - 4)
        shape = 0.00042 * np.sin(0.47 * seq + 0.29 * ordinal)
        return_value = shape + (early_slope if early else 0.0) + (late_slope if late else 0.0)
        price *= float(np.exp(return_value))

        trade_increment = float(8 + (ordinal % 6) + ((seq * (ordinal + 3)) % 7))
        if late:
            trade_increment += float((ordinal * 3) % 11)
        trades += trade_increment
        mean_trade_size = 1_000.0 + 37.0 * ordinal + 180.0 * np.sin(0.21 * seq + ordinal)
        if late:
            mean_trade_size *= 1.0 + 0.015 * ((ordinal % 9) - 4)
        amount += trade_increment * mean_trade_size

        base_depth = (1_200.0 + 19.0 * ordinal) * (1.0 + 0.005 * (ordinal - 18) * int(late))
        touch_shape = 0.72 + 0.018 * ordinal + 0.12 * np.sin(0.31 * seq + ordinal)
        imbalance = 0.35 * np.sin((0.17 + 0.01 * (ordinal % 5)) * seq + 0.43 * ordinal)
        row: dict[str, object] = {
            "dt": DT,
            "code": code,
            "seq": seq,
            "trade_time": DT.normalize() + clock,
            "last": price,
            "amount": amount,
            "num_trades": trades,
        }
        for level in range(1, 6):
            level_weight = touch_shape if level == 1 else 1.0 + 0.04 * level * np.cos(0.19 * seq + ordinal)
            level_depth = base_depth * level_weight / (0.7 + 0.25 * level)
            row[f"bid_volume{level}"] = level_depth * (1.0 + imbalance)
            row[f"ask_volume{level}"] = level_depth * (1.0 - imbalance)
        rows.append(row)
    return pd.DataFrame(rows).set_index(["dt", "code", "seq"])


def _stock_panel(*, count: int = len(STOCKS)) -> pd.DataFrame:
    panel = pd.concat(
        [_stock_path(code=code, ordinal=ordinal).reset_index() for ordinal, code in enumerate(STOCKS[:count])],
        ignore_index=True,
    ).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _bond_panel() -> pd.DataFrame:
    panel = pd.DataFrame(
        {
            "dt": [DT],
            "code": [BOND],
            "seq": [0],
            "trade_time": [DT.normalize() + pd.Timedelta(hours=14, minutes=25)],
        }
    ).set_index(["dt", "code", "seq"])
    panel.attrs["__build_day__"] = DT.date().isoformat()
    return panel


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=rank_state.KERNEL_NAME, params={"signal": entry.signal})
        for entry in rank_state.factor_mining_catalog()
    ]


def _mapping(*, stock_code: str = PRIMARY_STOCK, trade_date: object = DT.normalize()) -> pd.DataFrame:
    return pd.DataFrame({"code": [BOND], "stock_code": [stock_code], "trade_date": [trade_date]})


def _build(*, stock_panel: pd.DataFrame | None = None, mapping: pd.DataFrame | None = None) -> pd.DataFrame:
    return build_factor_frame(
        _bond_panel(),
        _specs(),
        stock_panel=_stock_panel() if stock_panel is None else stock_panel,
        bond_stock_map=_mapping() if mapping is None else mapping,
    )


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    row = frame.loc[frame["code"] == PRIMARY_STOCK].iloc[[0]].copy()
    prior = row.copy()
    prior["seq"] = 10_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 1.0
    prior["amount"] = 1e12
    prior["num_trades"] = 1e9

    preopen = row.copy()
    preopen["seq"] = 10_001
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["last"] = 1.0
    preopen["amount"] = 1e12
    preopen["num_trades"] = 1e9

    after_cutoff = row.copy()
    after_cutoff["seq"] = 10_002
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_cutoff["last"] = 1.0
    after_cutoff["amount"] = 1e12
    after_cutoff["num_trades"] = 1e9

    at_1430 = row.copy()
    at_1430["seq"] = 10_003
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    at_1430["last"] = 1.0
    at_1430["amount"] = 1e12
    at_1430["num_trades"] = 1e9

    out = pd.concat([frame, prior, preopen, after_cutoff, at_1430], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_rank_state_catalogue_registration_and_context_contract() -> None:
    entries = rank_state.factor_mining_catalog()
    requirements = infer_factor_context_requirements(_specs())

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "underlying_cross_sectional_price_path_state": 3,
        "underlying_cross_sectional_liquidity_shape_state": 3,
        "underlying_cross_sectional_depth_shape_state": 3,
    }
    assert FactorRegistry.get(rank_state.KERNEL_NAME) is rank_state.FactorMiningUnderlyingCohortIntradayRankStateV1
    assert set(rank_state.FORMULAS) == {entry.signal for entry in entries}
    assert rank_state.FactorMiningUnderlyingCohortIntradayRankStateV1.requires_stock_panel is True
    assert rank_state.FactorMiningUnderlyingCohortIntradayRankStateV1.requires_bond_stock_map is True
    assert rank_state.FactorMiningUnderlyingCohortIntradayRankStateV1.daily_requirements() == []
    assert requirements.stock_panel_required is True
    assert requirements.bond_stock_map_required is True
    assert requirements.daily_required is False


def test_rank_state_builds_nine_finite_inherited_stock_ranks_without_inf() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in rank_state.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert int(frame.notna().sum(axis=1).iloc[0]) == 9
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert ((frame.to_numpy(dtype="float64") >= 0.0) & (frame.to_numpy(dtype="float64") <= 1.0)).all()


def test_rank_state_inherits_mapped_stock_rank_and_rejects_future_mapping() -> None:
    primary = _build(mapping=_mapping(stock_code=PRIMARY_STOCK))
    other = _build(mapping=_mapping(stock_code=OTHER_STOCK))
    future = _build(mapping=_mapping(trade_date=DT.normalize() + pd.Timedelta(days=1)))

    assert primary.loc[(DT, BOND), "ucris_stock_price_path_efficiency_rank"] != pytest.approx(
        other.loc[(DT, BOND), "ucris_stock_price_path_efficiency_rank"]
    )
    assert future.isna().all(axis=None)
    assert not np.isinf(future.to_numpy(dtype="float64")).any()


def test_rank_state_ignores_nonphysical_preopen_prior_and_after_cutoff_rows() -> None:
    baseline = _build()
    contaminated = _build(stock_panel=_with_out_of_contract_rows(_stock_panel()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_rank_state_mapping_and_panel_gaps_fail_closed_without_inf() -> None:
    duplicate = pd.concat([_mapping(), _mapping(stock_code=OTHER_STOCK)], ignore_index=True)
    duplicate_result = _build(mapping=duplicate)

    missing_last = _stock_panel().drop(columns=["last"])
    missing_last.attrs["__build_day__"] = DT.date().isoformat()
    missing_result = _build(stock_panel=missing_last)

    sparse_result = _build(stock_panel=_stock_panel(count=8))

    assert duplicate_result.isna().all(axis=None)
    assert missing_result.isna().all(axis=None)
    assert sparse_result.isna().all(axis=None)
    assert not np.isinf(duplicate_result.to_numpy(dtype="float64")).any()
    assert not np.isinf(missing_result.to_numpy(dtype="float64")).any()
    assert not np.isinf(sparse_result.to_numpy(dtype="float64")).any()


def test_rank_state_rejects_unknown_signal() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        rank_state.FactorMiningUnderlyingCohortIntradayRankStateV1().compute(
            FactorComputeContext(panel=_bond_panel(), params={"signal": "ucris_unknown"})
        )
