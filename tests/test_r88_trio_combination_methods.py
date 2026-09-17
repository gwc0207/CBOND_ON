from __future__ import annotations

import numpy as np
import pandas as pd

from harness.tools import r88_trio_combination_methods as combo


def _base_returns(days: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    values = np.linspace(-0.002, 0.003, days)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "benchmark_return": values * 0.3,
            "P6": values + 0.0002,
            "P3": values + 0.0001,
            "Regsim": values,
        }
    )


def test_simplex_grid_is_complete_and_valid() -> None:
    weights = combo._simplex_grid(0.25)
    assert len(weights) == 15
    assert all(np.all(value >= 0.0) for value in weights)
    assert all(np.isclose(value.sum(), 1.0) for value in weights)


def test_scores_from_rank_weights_preserves_daily_simplex_contract() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    panel = pd.DataFrame(
        {
            "trade_date": [dates[0]] * 21 + [dates[1]] * 21,
            "code": [f"A{index:02d}" for index in range(21)] * 2,
            "rank_P6": np.linspace(0.1, 0.9, 42),
            "rank_P3": np.linspace(0.2, 0.8, 42),
            "rank_Regsim": np.linspace(0.3, 0.7, 42),
        }
    )
    weights = combo._weights_frame(dates, np.repeat(np.array([[0.2, 0.3, 0.5]]), 2, axis=0), method_id="synthetic")
    scores = combo._scores_from_rank_weights(panel, weights)
    assert len(scores) == len(panel)
    assert scores["score"].between(0.0, 1.0).all()
    assert scores.groupby("trade_date").size().eq(21).all()


def test_dynamic_weighting_uses_only_strictly_prior_history() -> None:
    base = _base_returns(80)
    schedule, audit = combo._dynamic_weights_from_returns(base, method_id="fixed_share_hedge120")
    first_non_warmup = audit.loc[~audit["warmup_fallback_equal"]].iloc[0]
    assert int(first_non_warmup["history_days"]) == combo.MIN_TRAIN_DAYS
    warmup = schedule.loc[schedule["trade_date"] < first_non_warmup["trade_date"], ["weight_P6", "weight_P3", "weight_Regsim"]]
    assert np.allclose(warmup.to_numpy(dtype=float), combo.EQUAL_WEIGHTS)
    assert np.allclose(schedule[["weight_P6", "weight_P3", "weight_Regsim"]].sum(axis=1), 1.0)


def test_cdar_is_a_loss_measure() -> None:
    calm = np.array([0.001, 0.001, 0.001, 0.001])
    drawdown = np.array([0.001, -0.03, 0.001, 0.001])
    assert combo._drawdown_cdar(drawdown) < combo._drawdown_cdar(calm)


def test_cross_sectional_stacker_falls_back_without_prior_window() -> None:
    dates = pd.bdate_range("2025-01-02", periods=4)
    panel_rows = []
    label_rows = []
    for day in dates:
        for index in range(21):
            panel_rows.append(
                {
                    "trade_date": day,
                    "code": f"A{index:02d}",
                    "rank_P6": 0.1 + index / 100.0,
                    "rank_P3": 0.2 + index / 100.0,
                    "rank_Regsim": 0.3 + index / 100.0,
                }
            )
            label_rows.append({"trade_date": day, "code": f"A{index:02d}", "label_return_net": index / 10_000.0})
    scores, audit = combo._cross_sectional_stacker(pd.DataFrame(panel_rows), pd.DataFrame(label_rows), method_id="nnls_stack120")
    assert len(scores) == len(panel_rows)
    assert audit["warmup_fallback_equal"].all()
    assert np.allclose(audit[["weight_P6", "weight_P3", "weight_Regsim"]].to_numpy(dtype=float), combo.EQUAL_WEIGHTS)


def test_family_validation_accepts_mcs_membership_scope_column(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(combo, "BOOTSTRAP_REPS", 10)
    monkeypatch.setattr(combo, "MCS_BOOTSTRAP_REPS", 10)
    dates = pd.bdate_range("2025-01-02", periods=300)
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0004, 0.01, size=(len(dates), 3))
    data = pd.DataFrame(
        {
            "trade_date": list(dates) * 3,
            "method_id": np.repeat(["a", "b", "reference"], len(dates)),
            "day_return": np.concatenate([returns[:, 0], returns[:, 1], returns[:, 2]]),
            "benchmark_return": 0.0,
        }
    )
    result = combo._family_validation(
        run_root=tmp_path,
        data=data,
        method_ids=["a", "b", "reference"],
        reference_id="reference",
        family="synthetic",
        include_static_pbo=True,
    )
    assert result["ledger_method_count"] == 3
    assert result["formal_unique_return_sequence_count"] == 3
    assert (tmp_path / "validation" / "synthetic" / "mcs_membership.csv").is_file()


def test_unique_return_representatives_collapse_exact_duplicate_series() -> None:
    dates = pd.bdate_range("2025-01-02", periods=40)
    values = np.linspace(-0.001, 0.002, len(dates))
    data = pd.DataFrame(
        {
            "trade_date": list(dates) * 3,
            "method_id": np.repeat(["reference", "equal", "duplicate_equal"], len(dates)),
            "day_return": np.concatenate([values * 0.8, values, values]),
            "benchmark_return": 0.0,
        }
    )
    representatives, audit = combo._unique_return_representatives(
        data,
        method_ids=["equal", "duplicate_equal", "reference"],
        reference_id="reference",
    )
    assert representatives == ["reference", "equal"]
    duplicate = audit.loc[audit["method_id"].eq("duplicate_equal")].iloc[0]
    assert duplicate["formal_representative_method_id"] == "equal"
    assert not bool(duplicate["is_formal_representative"])
