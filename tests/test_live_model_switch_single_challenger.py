from datetime import date
from pathlib import Path

import pandas as pd

from cbond_on.app.usecases import live_runtime
from cbond_on.infra.live.model_switch import decide_single_challenger_by_sharpe


def test_single_challenger_uses_source_score_without_rank_average(monkeypatch, tmp_path):
    source_scores = pd.DataFrame(
        {
            "trade_date": [date(2026, 7, 8), date(2026, 7, 8)],
            "code": ["113001", "113002"],
            "score": [2.0, -1.0],
        }
    )
    source_path = tmp_path / "regsim_scores"

    def fake_run_source_score(*, source_cfg, score_day, label_cutoff, paths_cfg):
        assert source_cfg["model_id"] == "regsim"
        assert source_cfg["config"] == "live/regsim"
        return "regsim", "Regsim", source_path, source_scores

    monkeypatch.setattr(live_runtime, "_run_switch_source_score", fake_run_source_score)

    model_id, score_path, score_df, details = live_runtime._build_switch_challenger_score(
        switch_cfg={
            "champion": {"name": "Champion"},
            "challenger": {
                "name": "Regsim",
                "model_id": "regsim",
                "kind": "single",
                "config": "live/regsim",
            },
        },
        current_model_id="champion",
        current_score_path=Path("champion_scores"),
        current_score_df=pd.DataFrame({"code": ["113001"], "score": [0.5]}),
        score_day=date(2026, 7, 8),
        label_cutoff=date(2026, 7, 7),
        paths_cfg={"results_root": str(tmp_path)},
    )

    assert model_id == "regsim"
    assert score_path == source_path
    assert score_df["score"].tolist() == [2.0, -1.0]
    assert details == [
        {
            "name": "Regsim",
            "model_id": "regsim",
            "score_path": str(source_path),
            "rows": 2,
        }
    ]


def test_multi_challenger_sharpe_selects_best_candidate(tmp_path):
    days = pd.date_range("2026-07-01", periods=5, freq="D")

    def write_history(name, returns):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"trade_date": days, "day_return": returns}).to_csv(path, index=False)
        return str(path)

    cfg = {
        "metric": "rolling_sharpe",
        "lookback_days": 5,
        "min_periods": 5,
        "threshold": 0.0,
        "champion": {
            "name": "Champion",
            "model_id": "champion",
            "return_path": write_history("champion", [0.001, 0.002, -0.001, 0.001, 0.0]),
        },
        "challengers": [
            {
                "name": "Rankavg",
                "model_id": "rankavg",
                "return_path": write_history("rankavg", [0.006, 0.004, 0.007, 0.005, 0.006]),
            },
            {
                "name": "Regsim",
                "model_id": "regsim",
                "return_path": write_history("regsim", [0.002, -0.001, 0.003, 0.001, 0.0]),
            },
        ],
    }

    decision = decide_single_challenger_by_sharpe(cfg, score_day=date(2026, 7, 8))

    assert decision.selected_model_id == "rankavg"
    assert decision.challenger_model_id == "rankavg"
    assert decision.challenger_score is not None
    assert len(decision.candidate_scores or []) == 3


def test_multi_challenger_builder_supports_rankavg_and_single(monkeypatch, tmp_path):
    source_scores = {
        "baseline": pd.DataFrame({"code": ["113001", "113002"], "score": [0.1, 0.4]}),
        "regsim": pd.DataFrame({"code": ["113001", "113002"], "score": [2.0, -1.0]}),
    }

    def fake_run_source_score(*, source_cfg, score_day, label_cutoff, paths_cfg):
        model_id = source_cfg["model_id"]
        return model_id, source_cfg["name"], tmp_path / model_id, source_scores[model_id]

    monkeypatch.setattr(live_runtime, "_run_switch_source_score", fake_run_source_score)

    results = live_runtime._build_switch_challenger_scores(
        switch_cfg={
            "champion": {"name": "Champion"},
            "challengers": [
                {
                    "name": "Rankavg",
                    "model_id": "rankavg",
                    "kind": "rankavg",
                    "score_output": str(tmp_path / "rankavg_out"),
                    "sources": [
                        {"name": "champion", "model_id": "champion", "config": "live/champion"},
                        {"name": "baseline", "model_id": "baseline", "config": "live/baseline"},
                    ],
                },
                {
                    "name": "Regsim",
                    "model_id": "regsim",
                    "kind": "single",
                    "config": "live/regsim",
                },
            ],
        },
        current_model_id="champion",
        current_score_path=tmp_path / "champion_scores",
        current_score_df=pd.DataFrame({"code": ["113001", "113002"], "score": [0.5, 0.2]}),
        score_day=date(2026, 7, 8),
        label_cutoff=date(2026, 7, 7),
        paths_cfg={"results_root": str(tmp_path)},
    )

    assert [item["model_id"] for item in results] == ["rankavg", "regsim"]
    assert len(results[0]["score_df"]) == 2
    assert results[1]["score_df"]["score"].tolist() == [2.0, -1.0]
