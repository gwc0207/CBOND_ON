from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_intraday_transmission_response_v1 as transmission
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


DT = pd.Timestamp("2026-07-30 14:30:00")
BOND = "110001.SH"
STOCK = "600001.SH"
OTHER_STOCK = "600002.SH"


def _times() -> list[pd.Timedelta]:
    morning = [pd.Timedelta(hours=9, minutes=30 + 5 * idx) for idx in range(24)]
    afternoon = [pd.Timedelta(hours=13, minutes=5 * idx) for idx in range(18)]
    return morning + afternoon


def _returns(*, shape: int) -> list[float]:
    base = np.array(
        [
            0.0000,
            0.0015,
            -0.0008,
            0.0030,
            -0.0025,
            0.0007,
            0.0040,
            -0.0012,
            -0.0035,
            0.0020,
            0.0009,
            -0.0045,
            0.0011,
            0.0027,
            -0.0017,
            0.0036,
            -0.0029,
            0.0018,
            0.0044,
            -0.0010,
            -0.0038,
            0.0024,
            0.0013,
            -0.0042,
            0.0031,
            -0.0019,
            0.0047,
            -0.0022,
            0.0014,
            -0.0031,
            0.0038,
            -0.0011,
            0.0026,
            -0.0040,
            0.0016,
            0.0033,
            -0.0027,
            0.0042,
            -0.0015,
            0.0021,
            -0.0034,
            0.0037,
        ],
        dtype="float64",
    )
    if shape == 0:
        return base.tolist()
    if shape == 1:
        return (0.55 * base + 0.35 * np.roll(base, 1) - 0.10 * np.roll(base, 2)).tolist()
    return (-0.75 * base + 0.15 * np.roll(base, 3)).tolist()


def _panel(*, code: str, shape: int) -> pd.DataFrame:
    price = 100.0 if code == BOND else 10.0
    rows: list[dict[str, object]] = []
    for seq, (clock, log_return) in enumerate(zip(_times(), _returns(shape=shape), strict=True)):
        price *= float(np.exp(log_return))
        rows.append(
            {
                "dt": DT,
                "code": code,
                "seq": seq,
                "trade_time": DT.normalize() + clock,
                "last": price,
            }
        )
    out = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _stock_panel() -> pd.DataFrame:
    out = pd.concat(
        [_panel(code=STOCK, shape=0).reset_index(), _panel(code=OTHER_STOCK, shape=2).reset_index()],
        ignore_index=True,
    ).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=transmission.KERNEL_NAME, params={"signal": entry.signal})
        for entry in transmission.factor_mining_catalog()
    ]


def _mapping(*, stock_code: str = STOCK, trade_date: object = DT.normalize()) -> pd.DataFrame:
    return pd.DataFrame({"code": [BOND], "stock_code": [stock_code], "trade_date": [trade_date]})


def _build(
    *,
    bond_panel: pd.DataFrame | None = None,
    stock_panel: pd.DataFrame | None = None,
    mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_factor_frame(
        bond_panel if bond_panel is not None else _panel(code=BOND, shape=1),
        _specs(),
        stock_panel=_stock_panel() if stock_panel is None else stock_panel,
        bond_stock_map=_mapping() if mapping is None else mapping,
    )


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    prior = frame.iloc[[0]].copy()
    prior["seq"] = 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["last"] = 1.0

    preopen = frame.iloc[[0]].copy()
    preopen["seq"] = 2_000
    preopen["trade_time"] = DT.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["last"] = 1.0

    after_cutoff = frame.iloc[[-1]].copy()
    after_cutoff["seq"] = 3_000
    after_cutoff["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_cutoff["last"] = 1.0

    at_1430 = frame.iloc[[-1]].copy()
    at_1430["seq"] = 3_001
    at_1430["trade_time"] = DT.normalize() + pd.Timedelta(hours=14, minutes=30)
    at_1430["last"] = 1.0

    out = pd.concat([prior, frame, preopen, after_cutoff, at_1430], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    out.attrs["__build_day__"] = DT.date().isoformat()
    return out


def test_transmission_catalogue_registration_and_context_contract() -> None:
    entries = transmission.factor_mining_catalog()
    requirements = infer_factor_context_requirements(_specs())

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "intraday_stock_shock_directional_response": 3,
        "intraday_stock_shock_relative_amplitude": 3,
        "intraday_stock_shock_bond_kinetics": 3,
    }
    assert FactorRegistry.get(transmission.KERNEL_NAME) is transmission.FactorMiningIntradayTransmissionResponseV1
    assert set(transmission.FORMULAS) == {entry.signal for entry in entries}
    assert transmission.FactorMiningIntradayTransmissionResponseV1.requires_stock_panel is True
    assert transmission.FactorMiningIntradayTransmissionResponseV1.requires_bond_stock_map is True
    assert transmission.FactorMiningIntradayTransmissionResponseV1.daily_requirements() == []
    assert requirements.stock_panel_required is True
    assert requirements.bond_stock_map_required is True
    assert requirements.daily_required is False


def test_transmission_kernel_builds_nine_finite_mapped_stock_signals_without_inf() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in transmission.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(DT, BOND)]
    assert int(frame.notna().sum(axis=1).iloc[0]) == 9
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert -1.0 <= frame.loc[(DT, BOND), "itr_stock_shock_same_bin_directional_agreement"] <= 1.0
    assert 0.0 <= frame.loc[(DT, BOND), "itr_stock_shock_bond_response_deferment_share"] <= 1.0


def test_transmission_uses_context_mapping_and_rejects_explicit_future_mapping() -> None:
    baseline = _build(mapping=_mapping(stock_code=STOCK))
    other = _build(mapping=_mapping(stock_code=OTHER_STOCK))
    future = _build(mapping=_mapping(stock_code=STOCK, trade_date=DT.normalize() + pd.Timedelta(days=1)))

    assert baseline.loc[(DT, BOND), "itr_stock_shock_relative_amplitude_mean"] != pytest.approx(
        other.loc[(DT, BOND), "itr_stock_shock_relative_amplitude_mean"]
    )
    assert future.isna().all(axis=None)
    assert not np.isinf(future.to_numpy(dtype="float64")).any()


def test_transmission_ignores_nonphysical_preopen_prior_and_after_cutoff_rows() -> None:
    baseline = _build()
    contaminated = _build(
        bond_panel=_with_out_of_contract_rows(_panel(code=BOND, shape=1)),
        stock_panel=_with_out_of_contract_rows(_stock_panel()),
    )

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_transmission_mapping_and_panel_gaps_fail_closed_without_inf() -> None:
    duplicate = pd.concat([_mapping(), _mapping(stock_code=OTHER_STOCK)], ignore_index=True)
    duplicate_result = _build(mapping=duplicate)

    missing_price = _panel(code=BOND, shape=1).drop(columns=["last"])
    missing_price.attrs["__build_day__"] = DT.date().isoformat()
    missing_result = _build(bond_panel=missing_price)

    assert duplicate_result.isna().all(axis=None)
    assert missing_result.isna().all(axis=None)
    assert not np.isinf(duplicate_result.to_numpy(dtype="float64")).any()
    assert not np.isinf(missing_result.to_numpy(dtype="float64")).any()


def test_transmission_rejects_unknown_signal() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        transmission.FactorMiningIntradayTransmissionResponseV1().compute(
            FactorComputeContext(panel=_panel(code=BOND, shape=1), params={"signal": "itr_unknown"})
        )
