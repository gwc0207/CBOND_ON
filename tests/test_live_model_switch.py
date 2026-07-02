from __future__ import annotations

from datetime import date

import pandas as pd

from cbond_on.infra.live.model_switch import (
    build_rank_average_scores,
    decide_single_challenger_by_regime,
    decide_single_challenger_by_sharpe,
)
from cbond_on.infra.live import shadow_returns


def test_build_rank_average_scores_uses_cross_sectional_pct_rank() -> None:
    left = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "score": [1.0, 2.0, 3.0],
        }
    )
    right = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "score": [3.0, 1.0, 2.0],
        }
    )

    out = build_rank_average_scores(
        [("left", left), ("right", right)],
        score_day=date(2026, 6, 30),
    ).set_index("code")

    assert out.loc["A", "score"] == (1 / 3 + 1.0) / 2
    assert out.loc["B", "score"] == (2 / 3 + 1 / 3) / 2
    assert out.loc["C", "score"] == (1.0 + 2 / 3) / 2


def test_decide_single_challenger_by_sharpe_excludes_score_day(tmp_path) -> None:
    champion_path = tmp_path / "champion.csv"
    challenger_path = tmp_path / "challenger.csv"
    pd.DataFrame(
        {
            "trade_date": ["2026-06-26", "2026-06-29", "2026-06-30"],
            "day_return": [0.0, 0.002, 0.50],
        }
    ).to_csv(champion_path, index=False)
    pd.DataFrame(
        {
            "trade_date": ["2026-06-26", "2026-06-29", "2026-06-30"],
            "day_return": [0.01, 0.012, -0.50],
        }
    ).to_csv(challenger_path, index=False)
    cfg = {
        "metric": "rolling_sharpe",
        "lookback_days": 2,
        "min_periods": 2,
        "threshold": 0.0,
        "champion": {
            "name": "champion",
            "model_id": "champion_model",
            "return_path": str(champion_path),
        },
        "challenger": {
            "name": "challenger",
            "model_id": "challenger_model",
            "return_path": str(challenger_path),
        },
    }

    decision = decide_single_challenger_by_sharpe(cfg, score_day=date(2026, 6, 30))

    assert decision.history_end == date(2026, 6, 29)
    assert decision.history_days == 2
    assert decision.selected_model_id == "challenger_model"


def test_decide_single_challenger_by_regime_uses_prior_benchmark_state(tmp_path) -> None:
    champion_path = tmp_path / "champion.csv"
    challenger_path = tmp_path / "challenger.csv"
    dates = [
        "2026-06-22",
        "2026-06-23",
        "2026-06-24",
        "2026-06-25",
        "2026-06-26",
        "2026-06-29",
        "2026-06-30",
    ]
    pd.DataFrame(
        {
            "trade_date": dates,
            "day_return": [0.0, 0.0, 0.02, 0.03, -0.02, 0.01, -0.50],
            "benchmark_return": [0.01, 0.01, 0.01, -0.03, 0.02, 0.02, -0.50],
        }
    ).to_csv(champion_path, index=False)
    pd.DataFrame(
        {
            "trade_date": dates,
            "day_return": [0.0, 0.0, 0.00, 0.00, 0.03, 0.00, 0.50],
        }
    ).to_csv(challenger_path, index=False)
    cfg = {
        "metric": "bm20_sign",
        "lookback_days": 5,
        "min_periods": 2,
        "threshold": 0.0,
        "benchmark_window_days": 2,
        "champion": {
            "name": "champion",
            "model_id": "champion_model",
            "return_path": str(champion_path),
        },
        "challenger": {
            "name": "challenger",
            "model_id": "challenger_model",
            "return_path": str(challenger_path),
        },
    }

    decision = decide_single_challenger_by_regime(cfg, score_day=date(2026, 6, 30))

    assert decision.history_end == date(2026, 6, 29)
    assert decision.regime_state == "up"
    assert decision.regime_observations == 2
    assert decision.selected_model_id == "champion_model"
    assert decision.reason == "regime_champion_default"


def test_decide_single_challenger_by_regime_falls_back_to_sharpe(tmp_path) -> None:
    champion_path = tmp_path / "champion.csv"
    challenger_path = tmp_path / "challenger.csv"
    pd.DataFrame(
        {
            "trade_date": ["2026-06-24", "2026-06-25", "2026-06-26", "2026-06-29"],
            "day_return": [0.0, 0.001, 0.002, 0.003],
            "benchmark_return": [0.01, 0.01, 0.01, 0.01],
        }
    ).to_csv(champion_path, index=False)
    pd.DataFrame(
        {
            "trade_date": ["2026-06-24", "2026-06-25", "2026-06-26", "2026-06-29"],
            "day_return": [0.01, 0.011, 0.012, 0.013],
        }
    ).to_csv(challenger_path, index=False)
    cfg = {
        "metric": "bm20_sign",
        "lookback_days": 4,
        "min_periods": 3,
        "benchmark_window_days": 2,
        "fallback": {
            "lookback_days": 2,
            "min_periods": 2,
        },
        "champion": {
            "name": "champion",
            "model_id": "champion_model",
            "return_path": str(champion_path),
        },
        "challenger": {
            "name": "challenger",
            "model_id": "challenger_model",
            "return_path": str(challenger_path),
        },
    }

    decision = decide_single_challenger_by_regime(cfg, score_day=date(2026, 6, 30))

    assert decision.selected_model_id == "challenger_model"
    assert decision.reason == "fallback_rolling_sharpe_insufficient_regime_observations"
    assert decision.fallback_reason == "challenger_sharpe_gt_champion"


def test_update_shadow_return_history_appends_after_warm_start(tmp_path, monkeypatch) -> None:
    return_path = tmp_path / "returns.csv"
    score_path = tmp_path / "scores"
    score_path.mkdir()
    pd.DataFrame(
        {
            "trade_date": ["2026-06-29"],
            "day_return": [0.01],
        }
    ).to_csv(return_path, index=False)

    calls: list[tuple[date, date]] = []

    def fake_build(**kwargs):
        calls.append((kwargs["start_day"], kwargs["end_day"]))
        return pd.DataFrame(
            {
                "trade_date": [date(2026, 6, 29), date(2026, 6, 30)],
                "day_return": [999.0, 0.02],
                "count": [20, 20],
            }
        )

    monkeypatch.setattr(shadow_returns, "_build_shadow_daily_returns", fake_build)

    result = shadow_returns.update_shadow_return_history(
        model_id="model_a",
        raw_data_root="raw",
        score_path=score_path,
        return_path=return_path,
        expected_history_end=date(2026, 6, 30),
        strategy_id="strategy01_topk_turnover",
        strategy_config={"top_k": 20},
        allowlist_cfg={"enabled": True},
    )

    written = pd.read_csv(return_path)

    assert calls == [(date(2026, 6, 29), date(2026, 6, 30))]
    assert result.status == "updated"
    assert result.appended_rows == 1
    assert list(written["trade_date"]) == ["2026-06-29", "2026-06-30"]
    assert list(written["day_return"]) == [0.01, 0.02]


def test_update_shadow_return_history_skips_when_warm_start_missing(tmp_path, monkeypatch) -> None:
    return_path = tmp_path / "returns.csv"
    score_path = tmp_path / "scores"
    score_path.mkdir()
    pd.DataFrame(
        {
            "trade_date": ["2026-06-29"],
            "day_return": [0.01],
        }
    ).to_csv(return_path, index=False)

    def fake_build(**kwargs):
        return pd.DataFrame(
            {
                "trade_date": [date(2026, 6, 30)],
                "day_return": [0.02],
            }
        )

    monkeypatch.setattr(shadow_returns, "_build_shadow_daily_returns", fake_build)

    result = shadow_returns.update_shadow_return_history(
        model_id="model_a",
        raw_data_root="raw",
        score_path=score_path,
        return_path=return_path,
        expected_history_end=date(2026, 6, 30),
        strategy_id="strategy01_topk_turnover",
        strategy_config={"top_k": 20},
        allowlist_cfg={"enabled": True},
    )

    written = pd.read_csv(return_path)

    assert result.status == "skipped"
    assert result.reason == "warm_start_score_missing"
    assert list(written["trade_date"]) == ["2026-06-29"]
