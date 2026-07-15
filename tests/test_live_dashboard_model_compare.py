from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from liveLaunch.web.app import (
    _build_model_overview,
    _collect_dashboard_trade_days,
    _read_model_return_history,
)


def _write_returns(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_date": ["2026-06-29", "2026-06-30", "2026-07-01"],
            "day_return": values,
            "benchmark_return": [0.001, 0.002, 0.003],
        }
    ).to_csv(path, index=False)


def _write_decision(path: Path, *, selected_model_id: str, selected_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": "scoreopt_t1430_dispersion",
                "metric": "lcb10",
                "score_day": "2026-07-01",
                "selected_model_id": selected_model_id,
                "selected_name": selected_name,
                "reason": "score_best",
                "threshold": 0.0003,
                "candidate_scores": [
                    {"role": "champion", "name": "HL20", "model_id": "hl20", "score": 0.001},
                    {"role": "challenger", "name": "Ensemble", "model_id": "ensemble", "score": 0.002},
                    {"role": "challenger", "name": "Regsim", "model_id": "regsim", "score": 0.0005},
                ],
                "similar_days": [
                    {
                        "trade_date": "2026-06-20",
                        "distance": 0.75,
                        "best_model_id": "ensemble",
                        "best_name": "Ensemble",
                        "model_returns": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_model_overview_uses_latest_target_for_duplicate_score_day(tmp_path) -> None:
    results_root = tmp_path / "results"
    live_root = results_root / "live"
    hl20_return = results_root / "analysis" / "hl20.csv"
    ensemble_return = results_root / "analysis" / "ensemble.csv"
    regsim_return = results_root / "analysis" / "regsim.csv"
    _write_returns(hl20_return, [0.01, 0.02, 0.01])
    _write_returns(ensemble_return, [0.02, 0.01, 0.03])
    _write_returns(regsim_return, [-0.01, 0.01, 0.02])

    _write_decision(
        live_root / "2026-07-02" / "model_switch_decision.json",
        selected_model_id="hl20",
        selected_name="HL20",
    )
    _write_decision(
        live_root / "2026-07-03" / "model_switch_decision.json",
        selected_model_id="ensemble",
        selected_name="Ensemble",
    )

    live_cfg = {
        "model_switch": {
            "mode": "scoreopt_t1430_dispersion",
            "metric": "lcb10",
            "return_col": "day_return",
            "lookback_days": 3,
            "nearest_k": 2,
            "champion": {
                "name": "HL20",
                "model_id": "hl20",
                "return_path": str(hl20_return),
            },
            "challengers": [
                {
                    "name": "Ensemble",
                    "model_id": "ensemble",
                    "return_path": str(ensemble_return),
                },
                {
                    "name": "Regsim",
                    "model_id": "regsim",
                    "return_path": str(regsim_return),
                },
            ],
        }
    }

    payload = _build_model_overview(
        live_cfg=live_cfg,
        results_root=results_root,
        live_root=live_root,
        day="2026-07-03",
        lookback=3,
    )

    assert len(payload["candidates"]) == 3
    assert len(payload["decisions"]) == 1
    assert payload["current_decision"]["target_day"] == "2026-07-03"
    assert payload["current_decision"]["selected_model_id"] == "ensemble"
    assert payload["current_decision"]["similar_days"][0]["trade_date"] == "2026-06-20"
    assert payload["selected_strategy"]["points"] == [
        {
            "trade_date": "2026-07-01",
            "model_id": "ensemble",
            "name": "Ensemble",
            "day_return": 0.03,
            "nav": 1.03,
        }
    ]


def test_dashboard_trade_days_exclude_weekends_and_future_targets(tmp_path) -> None:
    live_root = tmp_path / "live"
    for day in ("2026-07-10", "2026-07-11", "2026-07-15"):
        (live_root / day / "logs").mkdir(parents=True)

    days, current_day = _collect_dashboard_trade_days(
        live_root=live_root,
        current_day=date(2026, 7, 14),
        open_days=[date(2026, 7, 10), date(2026, 7, 14), date(2026, 7, 15)],
        trade_contexts=[
            {"buy_day": date(2026, 7, 10)},
            {"buy_day": date(2026, 7, 11)},
        ],
    )

    assert current_day == "2026-07-14"
    assert days == ["2026-07-14", "2026-07-10"]


def test_model_return_history_uses_trading_day_allowlist(tmp_path) -> None:
    path = tmp_path / "returns.csv"
    pd.DataFrame(
        {
            "trade_date": ["2026-07-10", "2026-07-11", "2026-07-13"],
            "day_return": [0.01, 0.50, -0.02],
        }
    ).to_csv(path, index=False)

    frame = _read_model_return_history(
        path,
        asof_day=date(2026, 7, 14),
        lookback=20,
        return_col="day_return",
        trading_days={date(2026, 7, 10), date(2026, 7, 13)},
    )

    assert frame["trade_date"].tolist() == [date(2026, 7, 10), date(2026, 7, 13)]
    assert frame["day_return"].tolist() == [0.01, -0.02]
