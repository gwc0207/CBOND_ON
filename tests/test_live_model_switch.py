from __future__ import annotations

from datetime import date

import pandas as pd

from cbond_on.infra.live.model_switch import (
    build_rank_average_scores,
    decide_scoreopt_bm_short,
    decide_scoreopt_t1430_dispersion,
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


def test_decide_scoreopt_bm_short_selects_best_candidate(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=18, freq="D")

    def write_history(name: str, returns: list[float], *, benchmark: list[float] | None = None) -> str:
        path = tmp_path / f"{name}.csv"
        frame = pd.DataFrame({"trade_date": dates, "day_return": returns})
        if benchmark is not None:
            frame["benchmark_return"] = benchmark
        frame.to_csv(path, index=False)
        return str(path)

    benchmark = [
        0.001,
        -0.001,
        0.002,
        -0.002,
        0.001,
        0.002,
        -0.001,
        0.003,
        -0.002,
        0.001,
        0.002,
        -0.001,
        0.001,
        -0.002,
        0.002,
        0.001,
        -0.001,
        0.002,
    ]
    cfg = {
        "mode": "scoreopt_bm_short",
        "metric": "trim20_lcb10",
        "lookback_days": 8,
        "nearest_k": 3,
        "min_periods": 3,
        "margin": 0.00005,
        "champion": {
            "name": "Champion",
            "model_id": "champion",
            "return_path": write_history("champion", [0.0] * len(dates), benchmark=benchmark),
        },
        "challengers": [
            {
                "name": "Rankavg",
                "model_id": "rankavg",
                "return_path": write_history("rankavg", [0.01] * len(dates)),
            },
            {
                "name": "Regsim",
                "model_id": "regsim",
                "return_path": write_history("regsim", [0.002] * len(dates)),
            },
        ],
    }

    decision = decide_scoreopt_bm_short(cfg, score_day=date(2026, 1, 19))

    assert decision.mode == "scoreopt_bm_short"
    assert decision.metric == "trim20_lcb10"
    assert decision.history_end == date(2026, 1, 18)
    assert decision.history_days == 3
    assert decision.selected_model_id == "rankavg"
    assert decision.reason == "score_best"
    assert len(decision.candidate_scores or []) == 3


def test_decide_scoreopt_bm_short_defaults_champion_when_history_insufficient(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=11, freq="D")
    champion_path = tmp_path / "champion.csv"
    challenger_path = tmp_path / "challenger.csv"
    pd.DataFrame(
        {
            "trade_date": dates,
            "day_return": [0.0] * len(dates),
            "benchmark_return": [0.001] * len(dates),
        }
    ).to_csv(champion_path, index=False)
    pd.DataFrame({"trade_date": dates, "day_return": [0.01] * len(dates)}).to_csv(challenger_path, index=False)
    cfg = {
        "mode": "scoreopt_bm_short",
        "metric": "trim20_lcb10",
        "lookback_days": 180,
        "nearest_k": 40,
        "min_periods": 40,
        "champion": {
            "name": "Champion",
            "model_id": "champion",
            "return_path": str(champion_path),
        },
        "challenger": {
            "name": "Challenger",
            "model_id": "challenger",
            "return_path": str(challenger_path),
        },
    }

    decision = decide_scoreopt_bm_short(cfg, score_day=date(2026, 1, 12))

    assert decision.selected_model_id == "champion"
    assert decision.reason == "insufficient_history"


def test_decide_scoreopt_t1430_dispersion_selects_nearest_best_candidate(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=7, freq="D")
    feature_cols = [
        "afternoon1300_1430_std",
        "afternoon1300_1430_iqr",
        "afternoon1300_1430_tail_spread",
        "last30_1330_1430_std",
        "last30_1330_1430_iqr",
        "last30_1330_1430_tail_spread",
        "dispersion_accel",
    ]
    feature_path = tmp_path / "state_features.csv"
    features = pd.DataFrame({"trade_date": dates})
    features["afternoon1300_1430_std"] = [-5.0, 1.0, 1.1, 1.2, -5.5, -6.0, 1.05]
    for col in feature_cols[1:]:
        features[col] = 0.0
    features.to_csv(feature_path, index=False)

    champion_path = tmp_path / "champion.csv"
    challenger_path = tmp_path / "challenger.csv"
    pd.DataFrame({"trade_date": dates[:-1], "day_return": [0.0] * 6}).to_csv(champion_path, index=False)
    pd.DataFrame(
        {
            "trade_date": dates[:-1],
            "day_return": [-0.01, 0.01, 0.012, 0.011, -0.01, -0.01],
        }
    ).to_csv(challenger_path, index=False)
    cfg = {
        "mode": "scoreopt_t1430_dispersion",
        "feature_set": "disp_afternoon7",
        "metric": "mean",
        "lookback_days": 6,
        "nearest_k": 3,
        "min_periods": 3,
        "margin": 0.0001,
        "state_feature_path": str(feature_path),
        "champion": {
            "name": "Champion",
            "model_id": "champion",
            "return_path": str(champion_path),
        },
        "challenger": {
            "name": "Challenger",
            "model_id": "challenger",
            "return_path": str(challenger_path),
        },
    }

    decision = decide_scoreopt_t1430_dispersion(cfg, score_day=date(2026, 1, 7))

    assert decision.mode == "scoreopt_t1430_dispersion"
    assert decision.history_end == date(2026, 1, 6)
    assert decision.history_days == 3
    assert decision.selected_model_id == "challenger"
    assert decision.reason == "score_best"


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
