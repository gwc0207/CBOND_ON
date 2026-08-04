from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cbond_on.app.usecases import backtest_runtime


def test_signal_day_lookup_uses_prior_open_day() -> None:
    friday = date(2026, 1, 2)
    monday = date(2026, 1, 5)
    tuesday = date(2026, 1, 6)

    assert backtest_runtime._signal_day_for_execution_index([friday, monday, tuesday], 1, 1) == friday
    assert backtest_runtime._signal_day_for_execution_index([friday, monday, tuesday], 0, 1) is None


def test_frozen_lagged_backtest_uses_prior_score_and_keeps_unfilled_weight_as_cash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    signal_day = date(2026, 1, 5)
    buy_day = date(2026, 1, 6)
    sell_day = date(2026, 1, 7)
    out_dir = tmp_path / "out"
    calls: dict[str, object] = {}
    cfg = {
        "start": str(buy_day),
        "end": str(buy_day),
        "batch_id": "lagged_test",
        "score_source": {"score_root": str(tmp_path / "scores")},
        "strategy_id": "strategy01_topk_turnover",
        "strategy_config": {},
        "execution_lag_trading_days": 1,
        "freeze_signal_universe": True,
        "allowlist": {
            "enabled": True,
            "table": "quant_factor_dev.researcher_xuvb.o_0005",
            "lag_trading_days": 1,
            "asset": "cbond",
        },
    }
    monkeypatch.setattr(
        backtest_runtime,
        "load_config_file",
        lambda key: {"raw_data_root": "raw", "results_root": str(tmp_path / "results")} if key == "paths" else {},
    )
    monkeypatch.setattr(backtest_runtime, "load_strategy_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(backtest_runtime, "resolve_score_path", lambda *args, **kwargs: tmp_path / "scores")
    monkeypatch.setattr(
        backtest_runtime,
        "load_scores_by_date",
        lambda *args, **kwargs: {
            signal_day: pd.DataFrame({"code": ["A", "B"], "score": [2.0, 1.0]})
        },
    )
    monkeypatch.setattr(backtest_runtime, "load_fees_buy_sell_bps", lambda: (1.0, 1.0, "test"))
    monkeypatch.setattr(backtest_runtime, "prev_trading_days_from_raw", lambda *args, **kwargs: [signal_day])
    monkeypatch.setattr(backtest_runtime, "iter_open_days", lambda *args, **kwargs: [buy_day])
    monkeypatch.setattr(backtest_runtime, "next_trading_days_from_raw", lambda *args, **kwargs: [sell_day])
    monkeypatch.setattr(backtest_runtime, "load_upstream_pool_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        backtest_runtime,
        "compute_benchmark_breakdowns_for_days",
        lambda **kwargs: pd.DataFrame(
            {
                "trade_date": [buy_day],
                "benchmark_return": [0.0],
                "buy_leg_ret_net": [0.0],
                "sell_leg_ret_net": [0.0],
                "buy_count": [2],
                "sell_count": [2],
                "fallback_sell_codes": [0],
                "fallback_sell_weight": [0.0],
                "benchmark_method": ["test"],
            }
        ),
    )
    monkeypatch.setattr(
        backtest_runtime,
        "load_strict_market_day",
        lambda **kwargs: pd.DataFrame(
            {
                "code": ["A", "B"],
                "buy_price": [100.0, 100.0],
                "buy_close_price": [100.0, 100.0],
                "buy_leg_ret_gross": [0.10, 0.0],
                "buy_leg_ret_net": [0.10, 0.0],
                # These mirror the execution-day sell helpers.  The IC path
                # must not pass them into the cycle builder because its
                # sell-day data would then collide with them.
                "sell_price": [100.0, 100.0],
                "strict_sell_leg_net_ret": [0.0, 0.0],
            }
        ),
    )

    def resolve_pool(**kwargs):
        calls["pool_reference_day"] = kwargs["trade_day"]
        return {"A", "B"}, {"fallback_no_filter": False, "allowlist_applied": True}

    monkeypatch.setattr(backtest_runtime, "resolve_pool_codes_for_trade_day", resolve_pool)

    def select(req):
        calls["selection_trade_date"] = req.trade_date
        calls["selection_codes"] = req.universe["code"].tolist()
        return pd.DataFrame(
            {"code": ["A", "B"], "score": [2.0, 1.0], "weight": [0.5, 0.5], "rank": [1, 2]}
        )

    monkeypatch.setattr(backtest_runtime, "select_signals", select)

    def build_holdings(**kwargs):
        calls["builder_buy_day"] = kwargs["buy_day"]
        calls["builder_normalize"] = kwargs["normalize"]
        # B is unavailable at execution.  Keeping A's original 50% weight
        # represents a 50% cash allocation rather than an after-the-fact 100% A bet.
        return kwargs["selection"].iloc[[0]].assign(
            buy_price=100.0,
            buy_close_price=100.0,
            buy_leg_ret_net=0.0,
            weighted_buy_leg_ret_net=0.0,
        )

    monkeypatch.setattr(backtest_runtime, "build_strict_buy_holdings_from_selection", build_holdings)

    def cycle(**kwargs):
        frame = kwargs["buy_holdings"].copy()
        calls.setdefault("cycle_input_columns", []).append(frame.columns.tolist())
        # The IC call has both scored names. Its mock strict sell leg is zero,
        # so the distinct buy legs are the full-cycle cross-sectional return.
        # The actual basket call has just A and still returns 10% on its 50%
        # filled weight, preserving the cash assertion below.
        if len(frame) == 2 and "buy_leg_ret_net" in frame.columns:
            frame["return_net"] = pd.to_numeric(frame["buy_leg_ret_net"], errors="coerce")
        else:
            frame["return_net"] = 0.10
        frame["weighted_return"] = frame["weight"] * frame["return_net"]
        frame["weighted_buy_leg_ret_net"] = 0.0
        frame["strict_sell_leg_net_ret"] = 0.10
        frame["weighted_sell_leg_ret_net"] = frame["weight"] * frame["strict_sell_leg_net_ret"]
        frame["sell_missing_fallback"] = False
        frame["sell_price"] = 110.0
        frame["full_cycle_ret_net"] = 0.10
        return frame

    monkeypatch.setattr(backtest_runtime, "compute_strict_cycle_detail_for_holdings", cycle)
    monkeypatch.setattr(backtest_runtime, "_build_output_dir", lambda *args, **kwargs: out_dir)
    monkeypatch.setattr(backtest_runtime, "write_backtest_report_image", lambda **kwargs: None)

    result = backtest_runtime.run(cfg=cfg)

    daily = pd.read_csv(result.out_dir / "daily_returns.csv")
    ic = pd.read_csv(result.out_dir / "ic.csv")
    assert calls["pool_reference_day"] == signal_day
    assert calls["selection_trade_date"] == signal_day
    assert calls["selection_codes"] == ["A", "B"]
    assert calls["builder_buy_day"] == buy_day
    assert calls["builder_normalize"] is False
    assert daily.loc[0, "score_day"] == signal_day.isoformat()
    assert daily.loc[0, "buy_day"] == buy_day.isoformat()
    assert daily.loc[0, "sell_day"] == sell_day.isoformat()
    assert daily.loc[0, "cash_weight"] == 0.5
    assert daily.loc[0, "day_return"] == 0.05
    assert ic.loc[0, "score_day"] == signal_day.isoformat()
    assert "buy_leg_ret_net" in calls["cycle_input_columns"][0]
    assert "sell_price" not in calls["cycle_input_columns"][0]
    assert ic.loc[0, "ic"] == pytest.approx(1.0)
    assert ic.loc[0, "rank_ic"] == pytest.approx(1.0)
