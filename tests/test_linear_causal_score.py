from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cbond_on.infra.model.adapters import LinearAdapter
from cbond_on.infra.model.impl.linear import linear_score
from cbond_on.infra.model.score_io import load_scores_by_date
from cbond_on.domain.factors.storage import FactorStore


_CODES = ["110001", "110002", "110003", "110004"]
_FACTOR_COLS = ["f1", "f2"]


def _write_factor_day(root: Path, day: date) -> None:
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    offset = float(day.day % 5)
    frame = pd.DataFrame(
        {
            "dt": [dt] * len(_CODES),
            "code": _CODES,
            "f1": [-2.0 + offset, -0.5 + offset, 0.5 + offset, 2.0 + offset],
            "f2": [1.5, -1.0, 0.25, -0.75],
        }
    ).set_index(["dt", "code"])
    path = root / "factors" / "T1430" / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _write_label_day(root: Path, day: date, *, scale: float = 1.0) -> None:
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=42)
    frame = pd.DataFrame(
        {
            "trade_time": [dt] * len(_CODES),
            "code": _CODES,
            "y": np.asarray([-0.03, -0.005, 0.015, 0.04]) * scale,
        }
    )
    path = root / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _run(
    *,
    factor_root: Path,
    label_root: Path,
    start: date,
    end: date,
    **overrides,
) -> linear_score.ScoreResult:
    kwargs = {
        "factor_root": factor_root,
        "factor_store": FactorStore(factor_root, panel_name="T1430"),
        "label_root": label_root,
        "start": start,
        "end": end,
        "factor_cols": _FACTOR_COLS,
        "panel_name": "T1430",
        "window_minutes": 15,
        "factor_time": "14:30",
        "label_time": "14:42",
        "min_count": 3,
        "winsor_lower": None,
        "winsor_upper": None,
        "zscore": True,
        "lookback_days": 3,
        "refit_freq": 1,
        "regression_alpha": 0.1,
        "weight_source": "regression",
        "fallback": "equal",
        "max_weight": 3.0,
        "normalize_weights": "l1",
        "manual_weights": pd.Series(0.0, index=_FACTOR_COLS),
        "device": "cpu",
        "gpu_fallback_to_cpu": True,
        "neutralizer": None,
        "audit_only": True,
    }
    kwargs.update(overrides)
    return linear_score.run_linear_score(**kwargs)


def _build_toy_history(tmp_path: Path) -> tuple[Path, Path, list[date]]:
    factor_root = tmp_path / "factor_data"
    label_root = tmp_path / "label_data"
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9)]
    for day in days:
        _write_factor_day(factor_root, day)
    for day in days[:4]:
        _write_label_day(label_root, day)
    return factor_root, label_root, days


def _scores(result: linear_score.ScoreResult) -> pd.DataFrame:
    return result.scores.sort_values(["trade_date", "code"]).reset_index(drop=True)


def test_target_score_never_reads_target_label_and_supports_missing_target_label(tmp_path, monkeypatch) -> None:
    factor_root, label_root, days = _build_toy_history(tmp_path)
    target = days[4]
    cutoff = days[3]
    real_read = linear_score._read_label_day

    def guarded_read(root, day, *, factor_time, label_time):
        if day == target:
            raise AssertionError("target label must not be read while scoring")
        return real_read(root, day, factor_time=factor_time, label_time=label_time)

    monkeypatch.setattr(linear_score, "_read_label_day", guarded_read)
    result = _run(
        factor_root=factor_root,
        label_root=label_root,
        start=target,
        end=target,
        label_cutoff=cutoff,
    )

    assert set(result.scores["code"]) == set(_CODES)
    assert pd.to_datetime(result.weights_history["train_end"]).dt.date.max() == cutoff
    assert (pd.to_datetime(result.weights_history["train_end"]).dt.date < target).all()


def test_injecting_target_label_does_not_change_target_score(tmp_path) -> None:
    factor_root, label_root, days = _build_toy_history(tmp_path)
    target = days[4]
    before = _run(factor_root=factor_root, label_root=label_root, start=target, end=target)
    _write_label_day(label_root, target, scale=-1_000.0)
    after = _run(factor_root=factor_root, label_root=label_root, start=target, end=target)

    pd.testing.assert_frame_equal(_scores(before), _scores(after), check_exact=True)
    pd.testing.assert_frame_equal(
        before.weights_history.reset_index(drop=True),
        after.weights_history.reset_index(drop=True),
        check_exact=True,
    )


def test_label_cutoff_excludes_later_training_labels(tmp_path, monkeypatch) -> None:
    factor_root, label_root, days = _build_toy_history(tmp_path)
    target = days[4]
    cutoff = days[2]
    calls: list[date] = []
    real_read = linear_score._read_label_day

    def guarded_read(root, day, *, factor_time, label_time):
        calls.append(day)
        if day > cutoff:
            raise AssertionError(f"future label was read: {day}")
        return real_read(root, day, factor_time=factor_time, label_time=label_time)

    monkeypatch.setattr(linear_score, "_read_label_day", guarded_read)
    result = _run(
        factor_root=factor_root,
        label_root=label_root,
        start=target,
        end=target,
        label_cutoff=cutoff,
    )

    assert calls
    assert max(calls) == cutoff
    assert pd.to_datetime(result.weights_history["train_end"]).dt.date.max() == cutoff


def test_all_linear_families_emit_finite_unique_scores(tmp_path) -> None:
    factor_root, label_root, days = _build_toy_history(tmp_path)
    target = days[4]
    variants = [
        {"regression_kind": "ridge", "regression_alpha": 0.1},
        {"regression_kind": "elasticnet", "regression_alpha": 0.0001, "elasticnet_l1_ratio": 0.5},
        {"regression_kind": "huber", "regression_alpha": 0.1, "huber_epsilon": 1.35},
    ]
    for variant in variants:
        result = _run(
            factor_root=factor_root,
            label_root=label_root,
            start=target,
            end=target,
            **variant,
        )
        assert len(result.scores) == len(_CODES)
        assert result.scores["code"].is_unique
        assert np.isfinite(result.scores["score"].to_numpy(dtype=float)).all()


def test_linear_adapter_forwards_label_cutoff(monkeypatch) -> None:
    from cbond_on.infra.model.runners import train_linear

    captured: dict = {}

    def fake_main(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(train_linear, "main", fake_main)
    LinearAdapter(Path("linear_config.json5")).predict(
        start="2026-07-28",
        end="2026-07-28",
        artifact=object(),
        label_cutoff="2026-07-27",
        execution={"refit_every_n_days": 1},
    )

    assert captured["label_cutoff"] == "2026-07-27"
    assert captured["start"] == "2026-07-28"
    assert captured["end"] == "2026-07-28"


def test_runner_with_label_cutoff_does_not_reopen_target_label(tmp_path, monkeypatch) -> None:
    from cbond_on.infra.model.runners import train_linear

    target = date(2026, 1, 9)
    cutoff = date(2026, 1, 8)
    sample = pd.DataFrame(
        {
            "dt": [pd.Timestamp(target) + pd.Timedelta(hours=14, minutes=30)] * len(_CODES),
            "code": _CODES,
            "f1": [-2.0, -0.5, 0.5, 2.0],
            "f2": [1.5, -1.0, 0.25, -0.75],
        }
    ).set_index(["dt", "code"])
    cfg = {
        "model_name": "runner_target_label_guard",
        "start": str(target),
        "end": str(target),
        "panel_name": "T1430",
        "window_minutes": 15,
        "factor_time": "14:30",
        "label_time": "14:42",
        "factors": _FACTOR_COLS,
        "zscore": True,
        "winsor": {"enabled": False},
        "min_count": 3,
        "bins": 5,
        "linear": {"lookback_days": 3, "refit_freq": 1, "weight_source": "regression"},
    }

    class FakeStore:
        def has_day(self, day: date) -> bool:
            assert day == target
            return True

        def read_day(self, day: date) -> pd.DataFrame:
            assert day == target
            return sample

    class FakeLogger:
        def log(self, *args, **kwargs) -> None:
            pass

        def finish(self, *args, **kwargs) -> None:
            pass

    def fake_load_config(name):
        if str(name) == "paths":
            return {
                "factor_data_root": str(tmp_path / "factor_data"),
                "label_data_root": str(tmp_path / "label_data"),
                "raw_data_root": str(tmp_path / "raw_data"),
                "panel_data_root": str(tmp_path / "panel_data"),
                "results_root": str(tmp_path / "results"),
            }
        return cfg

    score_result = linear_score.ScoreResult(
        scores=pd.DataFrame({"trade_date": [target] * len(_CODES), "code": _CODES, "score": [-1.0, -0.2, 0.3, 1.0]}),
        weights_history=pd.DataFrame(),
    )
    monkeypatch.setattr(train_linear, "load_config_file", fake_load_config)
    monkeypatch.setattr(train_linear, "build_factor_reader", lambda *args, **kwargs: FakeStore())
    monkeypatch.setattr(train_linear, "list_trading_days_from_raw", lambda *args, **kwargs: [target])
    monkeypatch.setattr(train_linear, "build_neutralizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_linear, "init_wandb_logger", lambda **kwargs: FakeLogger())
    monkeypatch.setattr(train_linear, "run_linear_score", lambda **kwargs: score_result)
    monkeypatch.setattr(
        train_linear,
        "_read_label_day",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("target label evaluation must be skipped")),
    )

    train_linear.main(
        config_path=tmp_path / "linear_config.json5",
        start=str(target),
        end=str(target),
        label_cutoff=str(cutoff),
    )


def test_non_overwrite_score_output_preserves_existing_days(tmp_path) -> None:
    score_root = tmp_path / "scores"
    weights_path = score_root / "weights.csv"
    first = linear_score.ScoreResult(
        scores=pd.DataFrame({"trade_date": [date(2026, 1, 5)], "code": ["110001"], "score": [0.1]}),
        weights_history=pd.DataFrame({"trade_date": [date(2026, 1, 5)], "factor": ["f1"], "weight": [0.1]}),
    )
    second = linear_score.ScoreResult(
        scores=pd.DataFrame({"trade_date": [date(2026, 1, 6)], "code": ["110002"], "score": [0.2]}),
        weights_history=pd.DataFrame({"trade_date": [date(2026, 1, 6)], "factor": ["f1"], "weight": [0.2]}),
    )
    replacement = linear_score.ScoreResult(
        scores=pd.DataFrame({"trade_date": [date(2026, 1, 6)], "code": ["110002"], "score": [0.3]}),
        weights_history=pd.DataFrame({"trade_date": [date(2026, 1, 6)], "factor": ["f1"], "weight": [0.3]}),
    )
    linear_score.write_linear_outputs(
        result=first,
        score_path=score_root,
        weights_path=weights_path,
        meta_path=None,
        meta_payload={},
        overwrite=True,
    )
    linear_score.write_linear_outputs(
        result=second,
        score_path=score_root,
        weights_path=weights_path,
        meta_path=None,
        meta_payload={},
        overwrite=False,
    )
    linear_score.write_linear_outputs(
        result=replacement,
        score_path=score_root,
        weights_path=weights_path,
        meta_path=None,
        meta_payload={},
        overwrite=False,
    )

    assert set(load_scores_by_date(score_root)) == {date(2026, 1, 5), date(2026, 1, 6)}
    weights = pd.read_csv(weights_path)
    assert len(weights) == 2
    assert weights.loc[weights["trade_date"] == "2026-01-06", "weight"].iloc[0] == 0.3


def test_research_elasticnet_warm_start_reuses_raw_checkpoint_parameters(tmp_path, monkeypatch) -> None:
    """Only explicit incremental ElasticNet uses the scratch state chain."""

    factor_root, label_root, days = _build_toy_history(tmp_path)
    state_dir = tmp_path / "research_results" / "model_state"
    captured: list[linear_score.LinearFitResult | None] = []

    def fake_fit(train_df, factor_cols, **kwargs):
        prior = kwargs.get("warm_start")
        captured.append(prior)
        # Deliberately use values which would change if score-normalised
        # weights, rather than raw parameters, were persisted.
        return linear_score.LinearFitResult(
            coef=pd.Series([2.0, -1.0], index=factor_cols, dtype=float),
            intercept=0.375,
        )

    monkeypatch.setattr(linear_score, "_fit_linear_parameters", fake_fit)
    result = _run(
        factor_root=factor_root,
        label_root=label_root,
        start=days[2],
        end=days[4],
        regression_kind="elasticnet",
        incremental_enabled=True,
        incremental_warm_start=True,
        incremental_save_state=True,
        state_dir=state_dir,
        warm_start_fingerprint="research-contract-v1",
    )

    assert not result.scores.empty
    assert len(captured) == 3
    assert captured[0] is None
    assert captured[1] is not None
    assert captured[1].coef.to_dict() == {"f1": 2.0, "f2": -1.0}
    assert captured[1].intercept == 0.375
    assert captured[2] is not None

    checkpoint = state_dir / f"{days[2]:%Y-%m-%d}.json"
    payload = __import__("json").loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["date"] == str(days[2])
    assert payload["train_end"] == str(days[1])
    assert payload["fingerprint"] == "research-contract-v1"
    assert payload["coef"] == [2.0, -1.0]
    assert payload["intercept"] == 0.375
    assert not list(state_dir.glob("*.tmp"))
    assert result.weights_history["warm_start_active"].all()


def test_elasticnet_warm_start_rejects_checkpoint_beyond_current_label_cutoff(tmp_path) -> None:
    state_path = tmp_path / "2026-01-07.json"
    fit = linear_score.LinearFitResult(
        coef=pd.Series([1.0, -2.0], index=_FACTOR_COLS, dtype=float),
        intercept=0.25,
    )
    linear_score._save_elasticnet_warm_start(
        checkpoint_path=state_path,
        fit=fit,
        factor_cols=_FACTOR_COLS,
        score_day=date(2026, 1, 7),
        train_end=date(2026, 1, 6),
        fingerprint="research-contract-v1",
        label_cutoff=date(2026, 1, 6),
    )

    rejected = linear_score._load_elasticnet_warm_start(
        checkpoint_path=state_path,
        expected_fingerprint="research-contract-v1",
        factor_cols=_FACTOR_COLS,
        target_day=date(2026, 1, 8),
        label_cutoff=date(2026, 1, 5),
    )
    assert rejected is None


def test_elasticnet_warm_start_rejects_payload_date_mismatch_with_filename(tmp_path) -> None:
    state_path = tmp_path / "2026-01-07.json"
    fit = linear_score.LinearFitResult(
        coef=pd.Series([1.0, -2.0], index=_FACTOR_COLS, dtype=float),
        intercept=0.25,
    )
    linear_score._save_elasticnet_warm_start(
        checkpoint_path=state_path,
        fit=fit,
        factor_cols=_FACTOR_COLS,
        score_day=date(2026, 1, 7),
        train_end=date(2026, 1, 6),
        fingerprint="research-contract-v1",
        label_cutoff=date(2026, 1, 6),
    )
    payload = __import__("json").loads(state_path.read_text(encoding="utf-8"))
    payload["date"] = "2026-01-06"
    state_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    assert linear_score._load_elasticnet_warm_start(
        checkpoint_path=state_path,
        expected_fingerprint="research-contract-v1",
        factor_cols=_FACTOR_COLS,
        target_day=date(2026, 1, 8),
        label_cutoff=date(2026, 1, 6),
    ) is None


def test_elasticnet_warm_start_copies_readonly_checkpoint_coefficients() -> None:
    train_df = pd.DataFrame(
        {
            "f1": [-2.0, -0.5, 0.5, 2.0, 3.0],
            "f2": [1.0, -1.0, 0.25, -0.5, 1.5],
            "y": [-0.04, -0.01, 0.01, 0.03, 0.05],
        }
    )
    readonly_coef = np.asarray([0.15, -0.05], dtype=float)
    readonly_coef.setflags(write=False)
    prior = linear_score.LinearFitResult(
        coef=pd.Series(readonly_coef, index=_FACTOR_COLS, dtype=float),
        intercept=0.01,
    )
    assert not prior.coef.to_numpy(copy=False).flags.writeable

    fitted = linear_score._fit_linear_parameters(
        train_df,
        _FACTOR_COLS,
        alpha=0.0001,
        regression_kind="elasticnet",
        elasticnet_l1_ratio=0.2,
        max_iter=2000,
        warm_start=prior,
    )

    assert fitted is not None
    assert np.isfinite(fitted.coef.to_numpy(dtype=float)).all()


def test_runner_requires_explicit_research_only_state_dir_for_elasticnet_warm_start(tmp_path) -> None:
    from cbond_on.infra.model.runners import train_linear

    class DummyNeutralizer:
        def summary(self) -> dict:
            return {"enabled": True, "name": "test"}

    results_root = tmp_path / "research_results"
    state_dir = results_root / "model_state" / "atomic_a"
    cfg = {
        "incremental": {
            "enabled": True,
            "warm_start": True,
            "save_state": True,
            "state_dir": str(state_dir),
        },
        "experiment": {"research_only": True},
    }
    kwargs = {
        "cfg": cfg,
        "paths_cfg": {"results_root": str(results_root)},
        "model_name": "atomic_a",
        "factor_cols": _FACTOR_COLS,
        "regression_kind": "elasticnet",
        "regression_alpha": 0.0001,
        "elasticnet_l1_ratio": 0.2,
        "max_iter": 2000,
        "lookback_days": 60,
        "refit_freq": 1,
        "factor_time": "14:30",
        "label_time": "14:42",
        "panel_name": "T1430",
        "winsor_lower": None,
        "winsor_upper": None,
        "zscore": True,
        "min_count": 12,
        "neutralizer": DummyNeutralizer(),
    }
    enabled, save_state, resolved_state_dir, fingerprint = train_linear._resolve_research_elasticnet_incremental(**kwargs)
    assert enabled is True
    assert save_state is True
    assert resolved_state_dir == state_dir
    assert isinstance(fingerprint, str) and len(fingerprint) == 64

    cfg["experiment"] = {"research_only": False}
    with __import__("pytest").raises(ValueError, match="research-only"):
        train_linear._resolve_research_elasticnet_incremental(**kwargs)

    cfg["experiment"] = {"research_only": True}
    cfg["incremental"]["state_dir"] = str(tmp_path / "outside_results")
    with __import__("pytest").raises(ValueError, match="inside paths.results_root"):
        train_linear._resolve_research_elasticnet_incremental(**kwargs)
