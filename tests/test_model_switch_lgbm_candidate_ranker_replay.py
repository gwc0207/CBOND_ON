from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from harness.tools import model_switch_lgbm_candidate_ranker_replay as replay


def _synthetic_panel(days: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = date(2025, 1, 2)
    for position in range(days):
        score_day = start + timedelta(days=position)
        returns = {
            "Regsim": 0.003 if position % 3 == 0 else 0.001,
            "Ensemble": 0.004 if position % 3 == 1 else 0.000,
            "HL20": 0.005 if position % 3 == 2 else -0.001,
        }
        for model_index, model in enumerate(replay.MODELS):
            relevance, rank = replay._relevance_and_rank(returns, model)
            row: dict[str, object] = {
                "score_day": score_day,
                "candidate": model,
                "execution_metadata_complete": True,
                "label_available_at": replay._label_available_timestamp(score_day + timedelta(days=1)),
                "relevance": relevance,
                "actual_rank": rank,
                "realized_return": returns[model],
            }
            row.update({f"realized_return_{name}": value for name, value in returns.items()})
            for feature_index, feature in enumerate(replay.GEOMETRY_FEATURES):
                row[feature] = float((position + 1) * (feature_index + 1) + model_index)
            for candidate_index, candidate in enumerate(replay.MODELS):
                row[f"candidate_is_{candidate}"] = float(model_index == candidate_index)
            rows.append(row)
    return pd.DataFrame(rows)


def _tiny_params() -> dict[str, object]:
    return {
        **replay.LGBM_PARAMS,
        "n_estimators": 4,
        "min_child_samples": 3,
        "min_split_gain": 0.0,
    }


def test_relevance_rank_preserves_ties() -> None:
    values = {"Regsim": 0.01, "Ensemble": 0.01, "HL20": -0.01}
    assert replay._relevance_and_rank(values, "Regsim") == (2, 1)
    assert replay._relevance_and_rank(values, "Ensemble") == (2, 1)
    assert replay._relevance_and_rank(values, "HL20") == (0, 3)


def test_direct_argmax_uses_fixed_candidate_order_for_exact_tie() -> None:
    assert replay._direct_argmax([1.0, 1.0, 0.0]) == 0
    assert replay._direct_argmax([0.0, 2.0, 2.0]) == 1


def test_expected_source_manifest_is_pinned() -> None:
    replay._assert_expected_source_manifest(
        {"source_manifest_sha256": replay.EXPECTED_SELECTOR_SOURCE_MANIFEST_SHA256}
    )
    with pytest.raises(RuntimeError, match="manifest SHA-256 drifted"):
        replay._assert_expected_source_manifest({"source_manifest_sha256": "not-the-frozen-source"})


def test_cli_accepts_explicit_120day_rolling_window() -> None:
    args = replay.parse_args(["--training-window-days", "120", "--min-training-days", "120"])
    assert args.training_window_days == 120
    assert args.min_training_days == 120


def test_lag_feature_excludes_label_not_mature_at_current_decision() -> None:
    days = [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)]
    returns = pd.DataFrame(
        {
            "score_day": days,
            "Regsim": [0.03, 0.01, 0.0],
            "Ensemble": [0.0, 0.0, 0.0],
            "HL20": [-0.03, -0.01, 0.0],
            "execution_metadata_complete": [True, True, True],
            "label_available_at": [
                pd.Timestamp("2025-01-04 15:00:00", tz="Asia/Shanghai"),
                pd.Timestamp("2025-01-04 09:39:00", tz="Asia/Shanghai"),
                pd.Timestamp("2025-01-05 09:39:00", tz="Asia/Shanghai"),
            ],
        }
    )
    lagged = replay._lag_relative_features(returns)
    day_three = lagged.loc[lagged["score_day"] == days[2]].iloc[0]
    assert day_three["lag_relative_ewm5_Regsim"] == pytest.approx(0.01)


def test_rolling_ranker_is_strictly_predecessor_and_future_label_invariant() -> None:
    panel = _synthetic_panel()
    original = replay.run_rolling_ranker(
        panel,
        variant="synthetic",
        feature_columns=replay.GEOMETRY_FEATURES,
        training_window_days=5,
        min_training_days=3,
        params=_tiny_params(),
    ).daily
    cutoff = pd.Timestamp("2025-01-08").date()
    poisoned = panel.copy()
    future = poisoned["score_day"] >= cutoff
    poisoned.loc[future, "realized_return"] *= -100.0
    poisoned.loc[future, "relevance"] = 2 - poisoned.loc[future, "relevance"].astype(int)
    rerun = replay.run_rolling_ranker(
        poisoned,
        variant="synthetic",
        feature_columns=replay.GEOMETRY_FEATURES,
        training_window_days=5,
        min_training_days=3,
        params=_tiny_params(),
    ).daily
    before = original.loc[original["score_day"] < cutoff].reset_index(drop=True)
    poisoned_before = rerun.loc[rerun["score_day"] < cutoff].reset_index(drop=True)
    assert before["selected_name"].tolist() == poisoned_before["selected_name"].tolist()
    assert np.allclose(
        before[[f"prediction_{model}" for model in replay.MODELS]].to_numpy(dtype=float),
        poisoned_before[[f"prediction_{model}" for model in replay.MODELS]].to_numpy(dtype=float),
        equal_nan=True,
    )
    ready = original.loc[original["prediction_ready"]]
    assert (pd.to_datetime(ready["training_end"]).dt.date < pd.to_datetime(ready["score_day"]).dt.date).all()
    assert (
        pd.to_datetime(ready["training_label_available_at_max"], utc=True)
        < pd.to_datetime(ready["decision_at"], utc=True)
    ).all()


def test_rolling_ranker_excludes_label_not_mature_at_decision() -> None:
    panel = _synthetic_panel()
    delayed_day = date(2025, 1, 5)
    panel.loc[panel["score_day"] == delayed_day, "label_available_at"] = pd.Timestamp(
        "2025-01-10 15:00:00", tz="Asia/Shanghai"
    )
    result = replay.run_rolling_ranker(
        panel,
        variant="synthetic",
        feature_columns=replay.GEOMETRY_FEATURES,
        training_window_days=5,
        min_training_days=3,
        params=_tiny_params(),
    ).daily
    ready = result.loc[result["prediction_ready"]]
    assert not ready.empty
    assert (
        pd.to_datetime(ready["training_label_available_at_max"], utc=True)
        < pd.to_datetime(ready["decision_at"], utc=True)
    ).all()


def test_score_csv_trade_date_mismatch_fails_closed(tmp_path) -> None:
    score_day = date(2025, 1, 2)
    for model in replay.MODELS:
        path = tmp_path / "scores" / model / "2025-01" / "2025-01-02.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "trade_date": ["2025-01-03", "2025-01-03", "2025-01-03"],
                "code": ["110001.SH", "110002.SH", "110003.SH"],
                "score": [3.0, 2.0, 1.0],
            }
        ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="trade_date"):
        replay._read_aligned_scores(tmp_path, score_day)


def test_frozen_factor_contract_rejects_order_drift(tmp_path) -> None:
    score_day = date(2025, 1, 2)
    columns = list(replay.FROZEN_LIVE50_FACTOR_CONTRACT)
    columns[0], columns[1] = columns[1], columns[0]
    index = pd.MultiIndex.from_tuples([(pd.Timestamp("2025-01-02 14:30:00"), "110001.SH")], names=["dt", "code"])
    frame = pd.DataFrame(np.ones((1, len(columns))), index=index, columns=columns)
    path = tmp_path / "factors" / "T1430" / "2025-01" / "20250102.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="schema/order"):
        replay._verify_frozen_factor_contract(tmp_path, [score_day])


def test_factor_feature_missing_value_is_not_silently_ready() -> None:
    panel = _synthetic_panel()
    panel["factor_probe"] = 1.0
    blocked_day = date(2025, 1, 7)
    panel.loc[panel["score_day"] == blocked_day, "factor_probe"] = np.nan
    result = replay.run_rolling_ranker(
        panel,
        variant="synthetic_factor",
        feature_columns=(*replay.GEOMETRY_FEATURES, "factor_probe"),
        training_window_days=5,
        min_training_days=3,
        params=_tiny_params(),
    ).daily
    blocked = result.loc[result["score_day"] == blocked_day].iloc[0]
    assert not bool(blocked["prediction_ready"])
    assert int(blocked["current_feature_missing_cells"]) == len(replay.MODELS)


def test_training_window_does_not_backfill_before_its_last_mature_days() -> None:
    panel = _synthetic_panel()
    panel["factor_probe"] = 1.0
    panel.loc[panel["score_day"] == date(2025, 1, 6), "factor_probe"] = np.nan
    result = replay.run_rolling_ranker(
        panel,
        variant="synthetic_window",
        feature_columns=(*replay.GEOMETRY_FEATURES, "factor_probe"),
        training_window_days=5,
        min_training_days=3,
        params=_tiny_params(),
    ).daily
    final_day = result.iloc[-1]
    assert int(final_day["mature_window_days"]) == 5
    assert int(final_day["training_days"]) == 4


def test_feature_panel_retains_label_maturity_fields(monkeypatch, tmp_path) -> None:
    codes = pd.Index([f"110{number:03d}.SH" for number in range(25)], name="code")
    factor_values = np.arange(len(codes) * len(replay.FROZEN_LIVE50_FACTOR_CONTRACT), dtype=float).reshape(
        len(codes), len(replay.FROZEN_LIVE50_FACTOR_CONTRACT)
    )

    def fake_scores(_input_root, _score_day):
        return pd.DataFrame(
            {
                "Regsim": np.arange(len(codes), 0, -1, dtype=float),
                "Ensemble": np.arange(len(codes), dtype=float),
                "HL20": np.linspace(0.0, 1.0, len(codes)),
            },
            index=codes,
        )

    def fake_factors(*, score_codes, factor_names, **_kwargs):
        return pd.DataFrame(factor_values, index=pd.Index(score_codes, name="code"), columns=list(factor_names))

    monkeypatch.setattr(replay, "_read_aligned_scores", fake_scores)
    monkeypatch.setattr(replay, "load_factor_cross_section", fake_factors)
    days = [date(2025, 1, 2), date(2025, 1, 3)]
    returns = pd.DataFrame(
        {
            "score_day": days,
            "Regsim": [0.001, 0.002],
            "Ensemble": [0.002, 0.001],
            "HL20": [0.0, 0.003],
            "execution_metadata_complete": [True, True],
            "buy_day": days,
            "sell_day": [day + timedelta(days=1) for day in days],
            "label_available_at": [replay._label_available_timestamp(day + timedelta(days=1)) for day in days],
        }
    )
    panel, _audit = replay.build_candidate_feature_panel(
        input_root=tmp_path,
        factor_root=tmp_path,
        returns=returns,
        factor_names=replay.FROZEN_LIVE50_FACTOR_CONTRACT,
    )
    assert {"buy_day", "sell_day", "label_available_at"}.issubset(panel.columns)
    assert panel["label_available_at"].notna().all()


def test_candidate_row_order_does_not_change_result() -> None:
    panel = _synthetic_panel()
    ordered = replay.run_rolling_ranker(
        panel,
        variant="synthetic",
        feature_columns=replay.GEOMETRY_FEATURES,
        training_window_days=5,
        min_training_days=3,
        params=_tiny_params(),
    ).daily
    shuffled = replay.run_rolling_ranker(
        panel.sample(frac=1.0, random_state=20260814).reset_index(drop=True),
        variant="synthetic",
        feature_columns=replay.GEOMETRY_FEATURES,
        training_window_days=5,
        min_training_days=3,
        params=_tiny_params(),
    ).daily
    assert ordered["selected_name"].tolist() == shuffled["selected_name"].tolist()
    assert np.allclose(
        ordered[[f"prediction_{model}" for model in replay.MODELS]].to_numpy(dtype=float),
        shuffled[[f"prediction_{model}" for model in replay.MODELS]].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_predeclared_factor_variant_is_bounded() -> None:
    assert set(replay.GEOMETRY_FEATURES).issubset(replay.VARIANTS["lgbm_geometry_factor_exposure"])
    assert len(replay.VARIANTS["lgbm_geometry_factor_exposure"]) <= 40


def test_results_writer_does_not_require_tabulate(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "variant": "synthetic",
                "scope": "full",
                "ready_only": True,
                "n_days": 3,
                "mean_rank": 1.5,
            }
        ]
    )
    importance = pd.DataFrame(
        [{"variant": "synthetic", "feature": "x", "gain": 1.0, "split": 1}]
    )
    replay._write_results(
        tmp_path,
        summary,
        importance,
        training_window_days=120,
        min_training_days=120,
    )
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "```csv" in text
    assert "synthetic" in text
    assert "max 120 strictly prior mature day groups" in text
