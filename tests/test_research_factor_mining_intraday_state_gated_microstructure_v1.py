from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.operators import research_factor_mining_intraday_state_gated_microstructure_v1 as gated
from cbond_on.domain.factors.spec import FactorSpec, infer_factor_context_requirements


SCORE = pd.Timestamp("2026-07-30 14:30:00")
CODES = tuple(f"{110001 + index:06d}.SH" for index in range(36))
BOND = CODES[25]


def _bare(code: str) -> str:
    return code.split(".", 1)[0]


def _times() -> list[pd.Timedelta]:
    morning = [pd.Timedelta(hours=9, minutes=30 + 5 * index) for index in range(24)]
    afternoon = [pd.Timedelta(hours=13, minutes=5 * index) for index in range(18)]
    return morning + afternoon


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mid = 100.0
    trades = 0.0
    amount = 0.0
    for seq, clock in enumerate(_times()):
        quote_revision = seq > 0 and seq % 3 == 0
        if quote_revision:
            mid *= float(np.exp(0.0004 * np.sin(0.41 * seq)))
        high_spread = (seq // 3) % 2 == 0
        spread = 0.08 * (1.8 if high_spread else 1.0)
        trade_increment = float(6 + (seq % 5))
        trades += trade_increment
        amount += trade_increment * (800.0 + 12.0 * (seq % 7))
        passive_wave = 1.0 + 0.13 * np.sin(0.73 * seq) + (0.06 if not quote_revision else -0.02)
        bid_depth = 1_100.0 * passive_wave * (1.0 + 0.05 * np.sin(0.31 * seq))
        ask_depth = 900.0 * passive_wave * (1.0 + 0.04 * np.cos(0.27 * seq))
        rows.append(
            {
                "dt": SCORE,
                "code": BOND,
                "seq": seq,
                "trade_time": SCORE.normalize() + clock,
                "last": mid,
                "amount": amount,
                "num_trades": trades,
                "bid_price1": mid - spread / 2.0,
                "ask_price1": mid + spread / 2.0,
                "bid_volume1": bid_depth,
                "ask_volume1": ask_depth,
            }
        )
    out = pd.DataFrame(rows).set_index(["dt", "code", "seq"])
    out.attrs["__build_day__"] = SCORE.date().isoformat()
    return out


def _daily_sources() -> dict[str, pd.DataFrame]:
    day = SCORE.normalize() - pd.offsets.BDay(1)
    price_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    ratings = ("CCC", "B", "BB", "BBB", "A", "AA-", "AA", "AA+", "AAA")
    for index, code in enumerate(CODES):
        price_rows.append(
            {
                "trade_date": day,
                "code": _bare(code),
                "exchange_code": "SH",
                "close_price": 100.0 + index,
            }
        )
        rating = "AAA" if code == BOND else ratings[index % len(ratings)]
        base_rows.append(
            {
                "trade_date": day,
                "code": _bare(code),
                "exchange_code": "SH",
                "duration": 1.0 + 0.12 * index,
                "rating": rating,
                "debt_puredebt_ratio": 75.0 + 1.4 * index,
                "bond_prem_ratio": 50.0 - 0.7 * index,
                "stock_volatility": 0.12 + 0.006 * index,
            }
        )
    return {
        "market_cbond.daily_price": pd.DataFrame(price_rows),
        "market_cbond.daily_base": pd.DataFrame(base_rows),
    }


def _specs() -> list[FactorSpec]:
    return [
        FactorSpec(name=entry.signal, factor=gated.KERNEL_NAME, params={"signal": entry.signal})
        for entry in gated.factor_mining_catalog()
    ]


def _build(
    sources: dict[str, pd.DataFrame] | None = None,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_factor_frame(panel if panel is not None else _panel(), _specs(), daily_data=_daily_sources() if sources is None else sources)


def _contaminate_daily(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source, frame in sources.items():
        terminal = frame.copy()
        score_day = terminal.copy()
        future = terminal.copy()
        score_day["trade_date"] = SCORE.normalize()
        future["trade_date"] = SCORE.normalize() + pd.offsets.BDay(1)
        for column in frame.columns:
            if column in {"trade_date", "code", "exchange_code", "rating"}:
                continue
            score_day[column] = 1e6
            future[column] = 2e6
        out[source] = pd.concat([frame, score_day, future], ignore_index=True)
    return out


def _with_out_of_contract_rows(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.reset_index()
    row = frame.iloc[[0]].copy()
    prior = row.copy()
    prior["seq"] = 1_000
    prior["trade_time"] = prior["trade_time"] - pd.Timedelta(days=1)
    prior["amount"] = 1e12
    prior["num_trades"] = 1e9

    preopen = row.copy()
    preopen["seq"] = 1_001
    preopen["trade_time"] = SCORE.normalize() + pd.Timedelta(hours=9, minutes=15)
    preopen["amount"] = 1e12
    preopen["num_trades"] = 1e9

    after_cutoff = row.copy()
    after_cutoff["seq"] = 1_002
    after_cutoff["trade_time"] = SCORE.normalize() + pd.Timedelta(hours=14, minutes=29, seconds=30)
    after_cutoff["amount"] = 1e12
    after_cutoff["num_trades"] = 1e9

    at_1430 = row.copy()
    at_1430["seq"] = 1_003
    at_1430["trade_time"] = SCORE.normalize() + pd.Timedelta(hours=14, minutes=30)
    at_1430["amount"] = 1e12
    at_1430["num_trades"] = 1e9

    out = pd.concat([frame, prior, preopen, after_cutoff, at_1430], ignore_index=True).set_index(
        ["dt", "code", "seq"]
    )
    out.attrs["__build_day__"] = SCORE.date().isoformat()
    return out


def test_gated_catalogue_registration_and_context_contract() -> None:
    entries = gated.factor_mining_catalog()
    requirements = gated.FactorMiningIntradayStateGatedMicrostructureV1.daily_requirements(
        {"signal": "isgm_duration_quote_revision_concentration"}
    )
    context_requirements = infer_factor_context_requirements(_specs())

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "duration_gated_quote_revision_calendar": 3,
        "credit_gated_spread_spell_topology": 3,
        "floor_premium_gated_passive_depth_refresh": 3,
        "stockvol_gated_trade_quote_clock_decoupling": 3,
    }
    assert FactorRegistry.get(gated.KERNEL_NAME) is gated.FactorMiningIntradayStateGatedMicrostructureV1
    assert set(gated.FORMULAS) == {entry.signal for entry in entries}
    assert [item.source for item in requirements] == ["market_cbond.daily_price", "market_cbond.daily_base"]
    assert requirements[0].columns == ("exchange_code", "close_price")
    assert requirements[1].columns == (
        "exchange_code",
        "duration",
        "rating",
        "debt_puredebt_ratio",
        "bond_prem_ratio",
        "stock_volatility",
    )
    assert context_requirements.stock_panel_required is False
    assert context_requirements.bond_stock_map_required is False
    assert context_requirements.daily_required is True


def test_gated_kernel_builds_twelve_finite_nonstatic_signals_without_inf() -> None:
    frame = _build()

    assert frame.columns.tolist() == [entry.signal for entry in gated.factor_mining_catalog()]
    assert frame.index.names == ["dt", "code"]
    assert frame.index.tolist() == [(SCORE, BOND)]
    assert int(frame.notna().sum(axis=1).iloc[0]) == 12
    assert not np.isinf(frame.to_numpy(dtype="float64")).any()
    assert frame.abs().sum(axis=1).iloc[0] > 0.0


def test_gated_kernel_ignores_score_day_and_future_daily_rows() -> None:
    baseline = _build()
    contaminated = _build(_contaminate_daily(_daily_sources()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_gated_kernel_ignores_nonphysical_panel_rows() -> None:
    baseline = _build()
    contaminated = _build(panel=_with_out_of_contract_rows(_panel()))

    pd.testing.assert_frame_equal(contaminated, baseline, check_exact=True)


def test_gated_kernel_stale_base_and_malformed_panel_fail_closed() -> None:
    sources = _daily_sources()
    anchor = sources["market_cbond.daily_price"]["trade_date"].max()
    stale_base = sources["market_cbond.daily_base"].loc[
        ~((sources["market_cbond.daily_base"]["code"] == _bare(BOND)) & (sources["market_cbond.daily_base"]["trade_date"] == anchor))
    ].copy()
    stale = _build({"market_cbond.daily_price": sources["market_cbond.daily_price"], "market_cbond.daily_base": stale_base})

    malformed = _panel().drop(columns=["bid_volume1"])
    malformed.attrs["__build_day__"] = SCORE.date().isoformat()
    malformed_result = _build(panel=malformed)

    assert stale.isna().all(axis=None)
    assert malformed_result.isna().all(axis=None)
    assert not np.isinf(stale.to_numpy(dtype="float64")).any()
    assert not np.isinf(malformed_result.to_numpy(dtype="float64")).any()


def test_gated_kernel_missing_source_duplicate_prior_and_unknown_signal_raise() -> None:
    kernel = gated.FactorMiningIntradayStateGatedMicrostructureV1()
    with pytest.raises(KeyError, match="missing daily source"):
        kernel.compute(FactorComputeContext(panel=_panel(), daily_data={}, params={"signal": _specs()[0].name}))

    sources = _daily_sources()
    duplicate_price = pd.concat(
        [sources["market_cbond.daily_price"], sources["market_cbond.daily_price"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate strict-prior rows"):
        kernel.compute(
            FactorComputeContext(
                panel=_panel(),
                daily_data={"market_cbond.daily_price": duplicate_price, "market_cbond.daily_base": sources["market_cbond.daily_base"]},
                params={"signal": _specs()[0].name},
            )
        )
    with pytest.raises(KeyError, match="unknown signal"):
        kernel.compute(FactorComputeContext(panel=_panel(), params={"signal": "isgm_unknown"}))
