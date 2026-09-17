from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_daily_debt_floor_wedge_dynamics_v1 as wedge
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = ("110001.SH", "110002.SH", "110003.SH")


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _panel() -> pd.DataFrame:
    rows = [
        {"dt": SCORE, "code": code, "seq": 0, "trade_time": SCORE, "last": 100.0}
        for code in CODES
    ]
    out = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = SCORE.date().isoformat()
    return out


def _daily_sources() -> dict[str, pd.DataFrame]:
    days = pd.bdate_range(end=SCORE.normalize() - pd.offsets.BDay(1), periods=66)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for code_index, code in enumerate(CODES):
        for index, day in enumerate(days):
            ytm = 0.021 + 0.00016 * index + 0.0017 * np.sin(0.23 * index + code_index)
            duration = 2.0 + 0.014 * index + 0.11 * code_index + 0.06 * np.cos(0.17 * index + code_index)
            residual = 0.75 * np.sin(0.49 * index + 0.7 * code_index) + 0.11 * np.cos(0.11 * index)
            debt_ratio = 82.0 + 0.8 * code_index + 68.0 * ytm + 0.72 * duration + residual
            pure_premium = 5.3 + 0.35 * code_index + 4.0 * np.sin(0.13 * index + 0.2 * code_index)
            price_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": "SH",
                    "close_price": 100.0 + code_index + 0.1 * index,
                }
            )
            base_rows.append(
                {
                    "trade_date": day,
                    "code": _bare(code),
                    "exchange_code": "SH",
                    "debt_puredebt_ratio": debt_ratio,
                    "puredebt_prem_ratio": pure_premium,
                    "ytm": ytm,
                    "duration": duration,
                }
            )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=wedge.KERNEL_NAME, params={"signal": entry.signal})
        for entry in wedge.factor_mining_catalog()
    ]


def _build(sources: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    return build_factor_frame(_panel(), _specs(), daily_data=_daily_sources() if sources is None else sources)


def _contaminate(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.groupby(["code", "exchange_code"], sort=False).tail(1).copy()
        score_day = terminal.copy()
        future = terminal.copy()
        score_day["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column in {"trade_date", "code", "exchange_code"}:
                continue
            score_day[column] = 1_000_000.0
            future[column] = 2_000_000.0
        out[source] = pd.concat([frame, score_day, future], ignore_index=True)
    return out


def test_wedge_catalogue_is_dynamic_and_has_explicit_context_contract() -> None:
    entries = wedge.factor_mining_catalog()
    requirements = wedge.FactorMiningDailyDebtFloorWedgeDynamicsV1.daily_requirements(
        {"signal": "dfw_wedge_delta1"}
    )
    context_requirements = infer_factor_context_requirements(_specs())

    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == {
        "debt_floor_wedge_change_acceleration": 3,
        "debt_floor_wedge_change_persistence": 3,
        "debt_floor_wedge_ytm_duration_innovation": 3,
    }
    assert "dfw_wedge_level" not in {entry.signal for entry in entries}
    assert FactorRegistry.get(wedge.KERNEL_NAME) is wedge.FactorMiningDailyDebtFloorWedgeDynamicsV1
    assert set(wedge.FORMULAS) == {entry.signal for entry in entries}
    assert [item.source for item in requirements] == ["market_cbond.daily_price", "market_cbond.daily_base"]
    assert requirements[0].columns == ("exchange_code", "close_price")
    assert requirements[1].columns == ("exchange_code", "debt_puredebt_ratio", "puredebt_prem_ratio", "ytm", "duration")
    assert all(item.lookback_days >= 66 for item in requirements)
    assert context_requirements.stock_panel_required is False
    assert context_requirements.bond_stock_map_required is False
    assert context_requirements.daily_required is True


def test_wedge_kernel_builds_nine_finite_strict_tminus1_signals_without_inf() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in wedge.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, code) for code in CODES]
    assert (frame.notna().sum(axis=0) == len(CODES)).all()
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame["dfw_wedge_delta1"].abs().sum() > 0.0
    assert frame["dfw_ytm_duration_oos_residual20"].abs().sum() > 0.0


def test_wedge_ignores_score_day_and_future_rows() -> None:
    baseline = _build()
    contaminated = _build(_contaminate(_daily_sources()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_wedge_stale_base_and_history_gap_fail_closed_on_exact_price_anchor() -> None:
    sources = _daily_sources()
    latest_day = sources["market_cbond.daily_price"]["trade_date"].max()
    stale_code = _bare(CODES[0])
    stale_base = sources["market_cbond.daily_base"].loc[
        ~((sources["market_cbond.daily_base"]["code"] == stale_code) & (sources["market_cbond.daily_base"]["trade_date"] == latest_day))
    ].copy()
    stale_frame = _build({"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": stale_base})

    gap_day = sorted(sources["market_cbond.daily_price"]["trade_date"].unique())[-2]
    gap_base = sources["market_cbond.daily_base"].loc[
        ~((sources["market_cbond.daily_base"]["code"] == _bare(CODES[1])) & (sources["market_cbond.daily_base"]["trade_date"] == gap_day))
    ].copy()
    gap_frame = _build({"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": gap_base})

    assert stale_frame.loc[(SCORE, CODES[0])].isna().all()
    assert stale_frame.loc[(SCORE, CODES[1])].notna().all()
    assert gap_frame.loc[(SCORE, CODES[1])].isna().all()
    assert not np.isinf(stale_frame.to_numpy(dtype="float64")).any()
    assert not np.isinf(gap_frame.to_numpy(dtype="float64")).any()


def test_wedge_missing_source_field_and_duplicate_strict_prior_rows_raise() -> None:
    kernel = wedge.FactorMiningDailyDebtFloorWedgeDynamicsV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(FactorComputeContext(panel=_panel(), daily_data={}, params={"signal": "dfw_wedge_delta1"}))

    sources = _daily_sources()
    missing_duration = sources["market_cbond.daily_base"].drop(columns=["duration"])
    with pytest.raises(KeyError, match="duration"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": missing_duration},
                params={"signal": "dfw_wedge_delta1"},
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
                daily_data={"market_cbond.daily_price": duplicate_price, "market_cbond.daily_base": sources["market_cbond.daily_base"]},
                params={"signal": "dfw_wedge_delta1"},
            )
        )


def test_wedge_rejects_unknown_signal() -> None:
    with pytest.raises(KeyError, match="unknown signal"):
        wedge.FactorMiningDailyDebtFloorWedgeDynamicsV1().compute(
            FactorComputeContext(panel=_panel(), params={"signal": "dfw_unknown"})
        )
