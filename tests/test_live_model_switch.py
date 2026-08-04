from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cbond_on.app.usecases import live_runtime
from cbond_on.infra.live import model_switch, shadow_returns
from cbond_on.infra.live.model_switch import (
    SwitchDecision,
    T1430_DISPERSION_FEATURE_SETS,
    _select_t1430_fusion,
    build_rank_average_scores,
    build_t1430_market_state_feature_row,
    decide_scoreopt_bm_short,
    decide_scoreopt_t1430_dispersion,
    decide_scoreopt_t1430_fusion_gate,
    decide_single_challenger_by_regime,
    decide_single_challenger_by_sharpe,
)


def _write_state_snapshot(root, *, score_day: date) -> None:
    rows: list[dict[str, object]] = []
    for code, multiplier in [("110001.SH", 1.0), ("110002.SH", 1.1)]:
        for stamp, price in [
            ("09:30:00", 100.0), ("09:31:00", 101.0),
            ("09:35:00", 102.0), ("09:36:00", 103.0),
            ("10:00:00", 104.0), ("10:01:00", 105.0),
            ("10:30:00", 106.0), ("10:31:00", 107.0),
            ("11:00:00", 108.0), ("11:01:00", 109.0),
            ("13:00:00", 110.0), ("13:01:00", 111.0),
            ("13:30:00", 112.0), ("13:31:00", 113.0),
            ("14:00:00", 114.0), ("14:28:00", 115.0),
            ("14:29:00", 116.0),
            # These observations are deliberately unavailable at strict 14:29.
            ("14:29:30", 1_000.0), ("14:30:00", 1_001.0),
        ]:
            rows.append(
                {
                    "code": code,
                    "trade_time": pd.Timestamp(f"{score_day} {stamp}"),
                    "last": price * multiplier,
                }
            )
    path = root / "snapshot" / "cbond" / f"{score_day:%Y-%m}" / f"{score_day:%Y%m%d}.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_market_state_builder_can_enforce_strict_1429_cutoff(tmp_path) -> None:
    score_day = date(2026, 7, 30)
    _write_state_snapshot(tmp_path, score_day=score_day)

    default = build_t1430_market_state_feature_row(clean_root=tmp_path, score_day=score_day)
    explicit_default = build_t1430_market_state_feature_row(
        clean_root=tmp_path,
        score_day=score_day,
        cutoff_time="14:30",
    )
    strict = build_t1430_market_state_feature_row(
        clean_root=tmp_path,
        score_day=score_day,
        cutoff_time="14:29",
    )

    assert default == explicit_default
    assert strict["full0935_1430_mean"] < default["full0935_1430_mean"]
    assert strict["seg1400_1430_mean"] < default["seg1400_1430_mean"]
    assert T1430_DISPERSION_FEATURE_SETS["path_full_t1429"] == T1430_DISPERSION_FEATURE_SETS["path_full_t1430"]


def test_market_state_builder_rejects_cutoff_after_live_boundary(tmp_path) -> None:
    score_day = date(2026, 7, 30)
    _write_state_snapshot(tmp_path, score_day=score_day)

    with pytest.raises(ValueError, match="within \\[14:00, 14:30\\]"):
        build_t1430_market_state_feature_row(
            clean_root=tmp_path,
            score_day=score_day,
            cutoff_time="14:31",
        )


def _switch_decision(
    *,
    reason: str,
    fallback_reason: str | None = None,
    fusion: dict | None = None,
) -> SwitchDecision:
    return SwitchDecision(
        enabled=True,
        mode="scoreopt_t1430_fusion_gate",
        metric="confidence_gate",
        lookback_days=60,
        min_periods=40,
        threshold=0.0005,
        score_day=date(2026, 7, 23),
        selected_model_id="champion",
        selected_name="Champion",
        champion_model_id="champion",
        champion_name="Champion",
        challenger_model_id="challenger",
        challenger_name="Challenger",
        champion_score=0.001,
        challenger_score=0.0008,
        score_diff=0.0002,
        history_end=date(2026, 7, 22),
        history_days=40,
        reason=reason,
        fallback_reason=fallback_reason,
        fusion=fusion,
    )


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
    assert decision.similar_days is not None
    assert len(decision.similar_days) == 3
    assert {item["trade_date"] for item in decision.similar_days} == {
        date(2026, 1, 2),
        date(2026, 1, 3),
        date(2026, 1, 4),
    }
    assert all(item["best_model_id"] == "challenger" for item in decision.similar_days)


def test_decide_scoreopt_t1430_dispersion_supports_full_path_feature_set(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=7, freq="D")
    feature_cols = T1430_DISPERSION_FEATURE_SETS["path_full_t1430"]
    assert "seg1000_1030_std" in feature_cols
    assert "seg1030_1100_tail_spread" in feature_cols
    assert "seg1100_1130_iqr" in feature_cols
    assert "seg1400_1430_pos_ratio" in feature_cols

    feature_path = tmp_path / "state_features.csv"
    features = pd.DataFrame({"trade_date": dates})
    for col in feature_cols:
        features[col] = 0.0
    features["seg1000_1030_std"] = [-5.0, 1.0, 1.1, 1.2, -5.5, -6.0, 1.05]
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
        "feature_set": "path_full_t1430",
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

    assert decision.selected_model_id == "challenger"
    assert decision.reason == "score_best"
    assert decision.history_days == 3


def test_decide_scoreopt_t1430_fusion_gate_overrides_low_confidence_base(tmp_path) -> None:
    dates = pd.date_range("2025-09-01", periods=101, freq="D")
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
    signal = pd.Series(range(len(dates)), dtype=float) / 50.0 - 1.0
    features = pd.DataFrame({"trade_date": dates})
    features[feature_cols[0]] = signal
    for col in feature_cols[1:]:
        features[col] = 0.0
    features.to_csv(feature_path, index=False)

    champion_path = tmp_path / "champion.csv"
    ensemble_path = tmp_path / "ensemble.csv"
    regsim_path = tmp_path / "regsim.csv"
    history_signal = signal.iloc[:-1].to_numpy()
    pd.DataFrame({"trade_date": dates[:-1], "day_return": 0.0}).to_csv(champion_path, index=False)
    pd.DataFrame(
        {"trade_date": dates[:-1], "day_return": 0.004 * history_signal}
    ).to_csv(ensemble_path, index=False)
    pd.DataFrame(
        {"trade_date": dates[:-1], "day_return": -0.004 * history_signal}
    ).to_csv(regsim_path, index=False)

    cfg = {
        "mode": "scoreopt_t1430_fusion_gate",
        "feature_set": "disp_afternoon7",
        "metric": "mean",
        "lookback_days": 100,
        "nearest_k": 20,
        "min_periods": 20,
        "margin": 0.1,
        "state_feature_path": str(feature_path),
        "champion": {
            "name": "Champion",
            "model_id": "champion",
            "return_path": str(champion_path),
        },
        "challengers": [
            {
                "name": "Ensemble",
                "model_id": "ensemble",
                "return_path": str(ensemble_path),
            },
            {
                "name": "Regsim",
                "model_id": "regsim",
                "return_path": str(regsim_path),
            },
        ],
        "fusion": {
            "policy": "base_low_confidence_only",
            "robust": {
                "lookback_days": 100,
                "min_periods": 80,
                "alpha": 0.0,
                "target_clip": 0.005,
                "margin": 0.0001,
            },
        },
    }

    decision = decide_scoreopt_t1430_fusion_gate(cfg, score_day=dates[-1].date())

    assert decision.mode == "scoreopt_t1430_fusion_gate"
    assert decision.selected_model_id == "ensemble"
    assert decision.reason == "fusion_robust_agrees_base_low_confidence"
    assert decision.fallback_reason == "margin_default"
    assert decision.fusion is not None
    assert decision.fusion["action"] == "robust_agrees_base_low_confidence"
    assert decision.fusion["robust"]["confident"] is True
    assert decision.fusion["robust"]["history_days"] == 100


def test_decide_scoreopt_t1430_fusion_gate_locks_narrow_champion_base_first(tmp_path) -> None:
    dates = pd.date_range("2025-09-01", periods=101, freq="D")
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
    signal = pd.Series(range(len(dates)), dtype=float) / 50.0 - 1.0
    features = pd.DataFrame({"trade_date": dates})
    features[feature_cols[0]] = signal
    for col in feature_cols[1:]:
        features[col] = 0.0
    features.to_csv(feature_path, index=False)

    champion_path = tmp_path / "champion.csv"
    ensemble_path = tmp_path / "ensemble.csv"
    hl20_path = tmp_path / "hl20.csv"
    history_signal = signal.iloc[:-1].to_numpy()
    pd.DataFrame(
        {"trade_date": dates[:-1], "day_return": [0.0] * 97 + [0.0041] * 3}
    ).to_csv(champion_path, index=False)
    pd.DataFrame(
        {"trade_date": dates[:-1], "day_return": 0.004 * history_signal}
    ).to_csv(ensemble_path, index=False)
    pd.DataFrame(
        {"trade_date": dates[:-1], "day_return": -0.004 * history_signal}
    ).to_csv(hl20_path, index=False)

    cfg = {
        "mode": "scoreopt_t1430_fusion_gate",
        "feature_set": "disp_afternoon7",
        "metric": "mean",
        "lookback_days": 100,
        "nearest_k": 3,
        "min_periods": 3,
        "margin": 0.0005,
        "state_feature_path": str(feature_path),
        "champion": {
            "name": "Champion",
            "model_id": "champion",
            "return_path": str(champion_path),
        },
        "challengers": [
            {
                "name": "Ensemble",
                "model_id": "ensemble",
                "return_path": str(ensemble_path),
            },
            {
                "name": "HL20",
                "model_id": "hl20",
                "return_path": str(hl20_path),
            },
        ],
        "fusion": {
            "policy": "base_low_confidence_only",
            "champion_first_override": {"enabled": False},
            "robust": {
                "lookback_days": 100,
                "min_periods": 80,
                "alpha": 0.0,
                "target_clip": 0.005,
                "margin": 0.0001,
            },
        },
    }

    baseline = decide_scoreopt_t1430_fusion_gate(cfg, score_day=dates[-1].date())

    assert baseline.selected_model_id == "ensemble"
    assert baseline.reason == "fusion_robust_override_base_low_confidence"
    assert baseline.fusion is not None
    assert baseline.fusion["base"]["selected_model_id"] == "champion"
    assert baseline.fusion["base"]["reason"] == "margin_default"
    assert 0.0 < baseline.fusion["base"]["score_diff"] < baseline.fusion["base"]["threshold"]
    assert baseline.fusion["robust"]["selected_model_id"] == "ensemble"
    assert baseline.fusion["robust"]["confident"] is True

    cfg["fusion"]["champion_first_override"]["enabled"] = True
    decision = decide_scoreopt_t1430_fusion_gate(cfg, score_day=dates[-1].date())

    assert decision.selected_model_id == "champion"
    assert decision.reason == "fusion_champion_base_first"
    assert decision.fallback_reason == "margin_default"
    assert decision.fusion is not None
    assert decision.fusion["action"] == "champion_base_first"
    assert decision.fusion["champion_first_override"]["triggered"] is True
    assert decision.fusion["robust"]["reason"] == "skipped_by_champion_base_first"
    assert live_runtime._model_switch_warnings(decision) == []


def test_decide_scoreopt_t1430_fusion_gate_champion_third_veto_skips_robust(tmp_path) -> None:
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
    ensemble_path = tmp_path / "ensemble.csv"
    hl20_path = tmp_path / "hl20.csv"
    pd.DataFrame({"trade_date": dates[:-1], "day_return": [-0.001] * 6}).to_csv(champion_path, index=False)
    pd.DataFrame({"trade_date": dates[:-1], "day_return": [0.0010] * 6}).to_csv(ensemble_path, index=False)
    pd.DataFrame({"trade_date": dates[:-1], "day_return": [0.0008] * 6}).to_csv(hl20_path, index=False)

    cfg = {
        "mode": "scoreopt_t1430_fusion_gate",
        "feature_set": "disp_afternoon7",
        "metric": "mean",
        "lookback_days": 6,
        "nearest_k": 3,
        "min_periods": 3,
        "margin": 0.01,
        "state_feature_path": str(feature_path),
        "champion": {
            "name": "Regsim",
            "model_id": "regsim",
            "return_path": str(champion_path),
        },
        "challengers": [
            {
                "name": "Ensemble",
                "model_id": "ensemble",
                "return_path": str(ensemble_path),
            },
            {
                "name": "HL20",
                "model_id": "hl20",
                "return_path": str(hl20_path),
            },
        ],
        "fusion": {
            "policy": "base_low_confidence_only",
            "champion_third_veto": {
                "enabled": True,
                "mode": "negative",
                "order": "before_robust_skip",
                "margin": 0.0005,
            },
            "robust": {
                "lookback_days": 100,
                "min_periods": 80,
                "alpha": 100.0,
                "target_clip": 0.0075,
                "margin": 0.0005,
            },
        },
    }

    decision = decide_scoreopt_t1430_fusion_gate(cfg, score_day=dates[-1].date())

    assert decision.selected_model_id == "ensemble"
    assert decision.reason == "fusion_champion_third_veto"
    assert decision.fusion is not None
    assert decision.fusion["action"] == "champion_third_veto"
    assert decision.fusion["champion_third_veto"]["triggered"] is True
    assert decision.fusion["robust"]["reason"] == "skipped_by_champion_third_veto"


def test_decide_scoreopt_t1430_fusion_gate_uses_base_best_when_robust_not_confident(tmp_path) -> None:
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
    ensemble_path = tmp_path / "ensemble.csv"
    hl20_path = tmp_path / "hl20.csv"
    pd.DataFrame({"trade_date": dates[:-1], "day_return": [0.0] * 6}).to_csv(champion_path, index=False)
    pd.DataFrame({"trade_date": dates[:-1], "day_return": [0.0010] * 6}).to_csv(ensemble_path, index=False)
    pd.DataFrame({"trade_date": dates[:-1], "day_return": [0.0008] * 6}).to_csv(hl20_path, index=False)

    cfg = {
        "mode": "scoreopt_t1430_fusion_gate",
        "feature_set": "disp_afternoon7",
        "metric": "mean",
        "lookback_days": 6,
        "nearest_k": 3,
        "min_periods": 3,
        "margin": 0.01,
        "state_feature_path": str(feature_path),
        "champion": {
            "name": "Regsim",
            "model_id": "regsim",
            "return_path": str(champion_path),
        },
        "challengers": [
            {
                "name": "Ensemble",
                "model_id": "ensemble",
                "return_path": str(ensemble_path),
            },
            {
                "name": "HL20",
                "model_id": "hl20",
                "return_path": str(hl20_path),
            },
        ],
        "fusion": {
            "policy": "base_low_confidence_only",
            "champion_third_veto": {
                "enabled": True,
                "mode": "negative",
                "order": "before_robust_skip",
                "margin": 0.0005,
            },
            "robust": {
                "lookback_days": 100,
                "min_periods": 80,
                "alpha": 100.0,
                "target_clip": 0.0075,
                "margin": 0.0005,
            },
        },
    }

    decision = decide_scoreopt_t1430_fusion_gate(cfg, score_day=dates[-1].date())

    assert decision.selected_model_id == "ensemble"
    assert decision.reason == "fusion_base_robust_not_confident"
    assert decision.fallback_reason == "margin_default"
    assert decision.fusion is not None
    assert decision.fusion["base"]["reason"] == "margin_default"
    assert decision.fusion["base"]["selected_model_id"] == "ensemble"
    assert decision.fusion["robust"]["reason"] == "insufficient_history"


@pytest.mark.parametrize(
    ("hl20_advantage", "expected_model_id", "expected_reason", "expected_veto_reason"),
    [
        (0.000733, "champion", "fusion_robust_veto_base_candidate", "base_robust_dominated_by_all"),
        (0.000100, "ensemble", "fusion_base_robust_not_confident", "base_not_dominated_by_all"),
    ],
)
def test_decide_scoreopt_t1430_fusion_gate_uses_robust_unanimous_base_veto(
    monkeypatch,
    hl20_advantage: float,
    expected_model_id: str,
    expected_reason: str,
    expected_veto_reason: str,
) -> None:
    base = SwitchDecision(
        enabled=True,
        mode="scoreopt_t1430_dispersion",
        metric="trim20_lcb10",
        lookback_days=60,
        min_periods=40,
        threshold=0.0005,
        score_day=date(2026, 7, 28),
        selected_model_id="ensemble",
        selected_name="Ensemble",
        champion_model_id="champion",
        champion_name="Champion",
        challenger_model_id="ensemble",
        challenger_name="Ensemble",
        champion_score=-0.000299,
        challenger_score=0.000008,
        score_diff=0.000307,
        history_end=date(2026, 7, 27),
        history_days=40,
        reason="margin_default",
        candidate_scores=[
            {"role": "champion", "name": "Champion", "model_id": "champion", "score": -0.000299},
            {"role": "challenger", "name": "Ensemble", "model_id": "ensemble", "score": 0.000008},
            {"role": "challenger", "name": "HL20", "model_id": "hl20", "score": -0.000365},
        ],
    )
    robust = {
        "reason": "margin_default",
        "confident": False,
        "selected_model_id": "champion",
        "selected_name": "Champion",
        "candidate_scores": [
            {"role": "champion", "name": "Champion", "model_id": "champion", "score": 0.000377},
            {"role": "challenger", "name": "Ensemble", "model_id": "ensemble", "score": -0.000546},
            {"role": "challenger", "name": "HL20", "model_id": "hl20", "score": 0.000169},
        ],
        "pairwise_predictions": [
            {
                "left_model_id": "champion",
                "left_name": "Champion",
                "right_model_id": "ensemble",
                "right_name": "Ensemble",
                "predicted_return_diff": 0.000905,
            },
            {
                "left_model_id": "champion",
                "left_name": "Champion",
                "right_model_id": "hl20",
                "right_name": "HL20",
                "predicted_return_diff": 0.000227,
            },
            {
                "left_model_id": "ensemble",
                "left_name": "Ensemble",
                "right_model_id": "hl20",
                "right_name": "HL20",
                "predicted_return_diff": -hl20_advantage,
            },
        ],
    }
    monkeypatch.setattr(model_switch, "decide_scoreopt_t1430_dispersion", lambda cfg, *, score_day: base)
    monkeypatch.setattr(model_switch, "_robust_pairwise_t1430_diagnostics", lambda cfg, *, score_day: robust)

    decision = decide_scoreopt_t1430_fusion_gate(
        {
            "fusion": {
                "policy": "base_low_confidence_only",
                "robust_base_veto": {
                    "enabled": True,
                    "margin": 0.0005,
                    "min_other_models": 2,
                    "base_low_confidence_only": True,
                },
            }
        },
        score_day=date(2026, 7, 28),
    )

    assert decision.selected_model_id == expected_model_id
    assert decision.reason == expected_reason
    assert decision.fusion is not None
    assert decision.fusion["robust_base_veto"]["reason"] == expected_veto_reason
    assert decision.fusion["robust_base_veto"]["triggered"] is (expected_model_id == "champion")


def test_live_model_switch_strict_records_margin_default_warning() -> None:
    decision = _switch_decision(reason="margin_default")

    live_runtime._assert_no_live_model_switch_fallback(
        decision,
        switch_cfg={"fail_on_fallback": True},
    )
    warnings = live_runtime._model_switch_warnings(decision)
    assert warnings
    assert warnings[0]["fallback_detail"] == "margin_default"
    assert warnings[0]["selected_model_id"] == "champion"


def test_live_model_switch_strict_records_unconfirmed_fusion_warning() -> None:
    decision = _switch_decision(
        reason="fusion_base_robust_not_confident",
        fallback_reason="margin_default",
        fusion={"base": {"reason": "margin_default"}},
    )

    live_runtime._assert_no_live_model_switch_fallback(
        decision,
        switch_cfg={"fail_on_fallback": True},
    )
    warnings = live_runtime._model_switch_warnings(decision)
    assert warnings
    assert warnings[0]["fallback_detail"] == "fusion_base_robust_not_confident:margin_default"
    assert warnings[0]["selected_model_id"] == "champion"


def test_live_model_switch_strict_allows_champion_third_veto() -> None:
    decision = _switch_decision(
        reason="fusion_champion_third_veto",
        fallback_reason="margin_default",
        fusion={
            "action": "champion_third_veto",
            "base": {"reason": "margin_default"},
        },
    )

    live_runtime._assert_no_live_model_switch_fallback(
        decision,
        switch_cfg={"fail_on_fallback": True},
    )


def test_select_t1430_fusion_keeps_high_confidence_base() -> None:
    base = SwitchDecision(
        enabled=True,
        mode="scoreopt_t1430_dispersion",
        metric="lcb10",
        lookback_days=120,
        min_periods=20,
        threshold=0.0003,
        score_day=date(2026, 7, 15),
        selected_model_id="champion",
        selected_name="Champion",
        champion_model_id="champion",
        champion_name="Champion",
        challenger_model_id="ensemble",
        challenger_name="Ensemble",
        champion_score=0.001,
        challenger_score=0.0005,
        score_diff=0.0005,
        history_end=date(2026, 7, 14),
        history_days=20,
        reason="score_best",
    )
    robust = {
        "confident": True,
        "selected_model_id": "ensemble",
        "selected_name": "Ensemble",
    }

    selected_model_id, _, reason, action = _select_t1430_fusion(
        base,
        robust,
        policy="base_low_confidence_only",
    )

    assert selected_model_id == "champion"
    assert reason == "fusion_base_high_confidence"
    assert action == "base_high_confidence"


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
