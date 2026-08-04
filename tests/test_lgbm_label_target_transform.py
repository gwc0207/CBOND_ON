from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cbond_on.infra.model.impl.lgbm import trainer as trainer_module
from cbond_on.infra.model.impl.lgbm.trainer import (
    LabelTargetTransformSpec,
    SplitData,
    _build_day_group_indices,
    _mean_pearson_ic_by_groups,
    prepare_lgbm_target_split,
    transform_label_targets_by_day,
    train_lgbm as fit_lgbm,
)
from cbond_on.infra.model.runners import train_lgbm


def test_zscore_day_target_transform_is_independent_for_each_completed_day() -> None:
    day_one = pd.Timestamp("2026-01-05 14:30")
    day_two = pd.Timestamp("2026-01-06 14:30")
    raw = pd.Series([1.0, 2.0, 3.0, 100.0, 120.0, 140.0])
    dt = pd.Series([day_one] * 3 + [day_two] * 3)

    transformed, audit = transform_label_targets_by_day(
        raw,
        dt,
        LabelTargetTransformSpec(mode="zscore_day", ddof=0),
    )

    assert raw.tolist() == [1.0, 2.0, 3.0, 100.0, 120.0, 140.0]
    grouped = pd.DataFrame({"day": dt.dt.normalize(), "target": transformed}).groupby("day")
    assert grouped["target"].mean().abs().max() < 1e-12
    assert np.allclose(grouped["target"].std(ddof=0).to_numpy(), [1.0, 1.0])
    assert audit["mode"] == "zscore_day"
    assert audit["day_count"] == 2


def test_zscore_day_target_transform_marks_constant_completed_day_for_drop() -> None:
    raw = pd.Series([0.01, 0.01, 0.01, 0.01, 0.02, 0.03])
    dt = pd.Series(
        [pd.Timestamp("2026-01-05 14:30")] * 3
        + [pd.Timestamp("2026-01-06 14:30")] * 3
    )

    transformed, audit = transform_label_targets_by_day(
        raw,
        dt,
        LabelTargetTransformSpec(mode="zscore_day"),
    )

    assert transformed.iloc[:3].isna().all()
    assert transformed.iloc[3:].notna().all()
    assert audit["dropped_day_count"] == 1
    assert audit["dropped_row_count"] == 3


def test_equal_day_loss_mass_preserves_within_day_ratios_and_global_scale() -> None:
    day_one = pd.Timestamp("2026-01-05 14:30")
    day_two = pd.Timestamp("2026-01-06 14:30")
    split = SplitData(
        x=pd.DataFrame({"factor": range(5)}),
        y=pd.Series([1.0, 2.0, 3.0, 4.0, 6.0]),
        dt=pd.Series([day_one] * 3 + [day_two] * 2),
        code=pd.Series(["a", "b", "c", "d", "e"]),
        sample_weight=pd.Series([1.0, 2.0, 3.0, 4.0, 8.0]),
    )

    prepared, audit = prepare_lgbm_target_split(
        split,
        LabelTargetTransformSpec(mode="zscore_day", loss_day_mass="equal"),
        apply_loss_day_mass=True,
    )

    masses = prepared.sample_weight.groupby(prepared.dt.dt.normalize()).sum()
    assert np.allclose(masses.to_numpy(), [2.5, 2.5])
    assert np.isclose(prepared.sample_weight.mean(), 1.0)
    assert np.isclose(prepared.sample_weight.iloc[1] / prepared.sample_weight.iloc[0], 2.0)
    assert audit["loss_day_mass"] == "equal"
    assert audit["loss_day_mass_applied"] is True


def test_daily_pearson_metric_is_equal_weighted_by_day() -> None:
    dt = pd.Series(
        [pd.Timestamp("2026-01-05")] * 3 + [pd.Timestamp("2026-01-06")] * 3
    )
    y = np.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    prediction = np.asarray([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])

    assert _mean_pearson_ic_by_groups(y, prediction, _build_day_group_indices(dt)) == 0.0


def test_target_transform_and_early_stopping_configs_are_opt_in() -> None:
    default_spec = train_lgbm._resolve_label_target_transform_spec({})
    assert default_spec.mode == "none"
    assert train_lgbm._resolve_early_stopping_metric({}, loss_mode="mse") == "rank_ic"

    spec = train_lgbm._resolve_label_target_transform_spec(
        {
            "label_target_transform": {
                "enabled": True,
                "mode": "zscore_day",
                "ddof": 0,
                "loss_day_mass": "equal",
            }
        }
    )
    assert spec.mode == "zscore_day"
    assert spec.loss_day_mass == "equal"
    assert train_lgbm._resolve_early_stopping_metric(
        {"early_stopping_metric": "pearson_ic"},
        loss_mode="mse",
    ) == "pearson_ic"
    assert train_lgbm._score_cache_may_read_target_label(
        label_anchor_lag_trading_days=0,
        score_only_no_target_label_read=True,
    ) is False
    assert train_lgbm._score_cache_may_read_target_label(
        label_anchor_lag_trading_days=0,
        score_only_no_target_label_read=False,
    ) is True
    assert train_lgbm._resolve_score_only_apply_tradable_filter({}) is False
    assert train_lgbm._resolve_score_only_apply_tradable_filter(
        {"score_only_apply_tradable_filter": True}
    ) is True


def test_score_only_can_apply_existing_allowlist_without_reading_target_label(monkeypatch) -> None:
    factor_day = date(2026, 1, 5)
    score_dt = pd.Timestamp("2026-01-05 14:30:00")

    class _FactorStore:
        def read_day(self, requested_day: date) -> pd.DataFrame:
            assert requested_day == factor_day
            return pd.DataFrame(
                {"factor": [1.0, 2.0, 3.0]},
                index=pd.MultiIndex.from_arrays(
                    [[score_dt] * 3, ["keep_a", "drop_me", "keep_b"]],
                    names=["dt", "code"],
                ),
            )

    def _unexpected_label_read(*_args, **_kwargs):
        pytest.fail("score-only dataset must not read the score-day label")

    monkeypatch.setattr(trainer_module, "_read_label_day", _unexpected_label_read)
    common = {
        "factor_store": _FactorStore(),
        "label_root": Path("unused"),
        "days": [factor_day],
        "factor_cols": ["factor"],
        "min_count": 2,
        "winsor_lower": None,
        "winsor_upper": None,
        "zscore": False,
        "factor_time": "14:30",
        "label_time": "14:42",
        "require_label": False,
        "tradable_code_map": {factor_day: {"keep_a", "keep_b"}},
        "tradable_strict": True,
        "read_label_when_not_required": False,
    }

    legacy = trainer_module.build_dataset(**common)
    filtered = trainer_module.build_dataset(
        **common,
        apply_tradable_filter_when_label_not_required=True,
    )

    assert set(legacy.code) == {"keep_a", "drop_me", "keep_b"}
    assert set(filtered.code) == {"keep_a", "keep_b"}
    assert filtered.y.isna().all()


def test_pearson_early_stop_uses_validation_only_when_split_sizes_match(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _CaptureRegressor:
        def __init__(self, **_params) -> None:
            pass

        def fit(self, _x, _y, **kwargs):
            eval_set = kwargs["eval_set"]
            captured["eval_set"] = eval_set
            eval_y = np.asarray(eval_set[0][1], dtype=float)
            prediction = np.asarray([1.0, 2.0, 3.0, 6.0, 5.0, 4.0])
            captured["metric"] = kwargs["eval_metric"](eval_y, prediction)
            captured["prediction"] = prediction
            return self

    monkeypatch.setattr(
        trainer_module,
        "lgb",
        SimpleNamespace(LGBMRegressor=_CaptureRegressor),
    )
    train = SplitData(
        x=pd.DataFrame({"f1": range(6)}),
        y=pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
        dt=pd.Series(
            [pd.Timestamp("2026-01-05 14:30")] * 3
            + [pd.Timestamp("2026-01-06 14:30")] * 3
        ),
        code=pd.Series([f"train_{i}" for i in range(6)]),
    )
    val = SplitData(
        x=pd.DataFrame({"f1": range(6)}),
        y=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        dt=pd.Series([pd.Timestamp("2026-01-07 14:30")] * 6),
        code=pd.Series([f"val_{i}" for i in range(6)]),
    )

    fit_lgbm(
        train=train,
        val=val,
        lgbm_params={"objective": "regression", "verbosity": -1, "device": "cpu"},
        early_stopping_rounds=2,
        label_target_transform=LabelTargetTransformSpec(
            mode="zscore_day",
            loss_day_mass="equal",
        ),
        early_stopping_metric="pearson_ic",
    )

    eval_set = captured["eval_set"]
    assert len(eval_set) == 1
    assert eval_set[0][0].equals(val.x)
    eval_y = np.asarray(eval_set[0][1], dtype=float)
    expected = trainer_module._mean_pearson_ic_by_groups(
        eval_y,
        captured["prediction"],
        trainer_module._build_day_group_indices(val.dt),
    )
    metric_name, actual, higher_is_better = captured["metric"]
    assert metric_name == "pearson_ic"
    assert np.isclose(actual, expected)
    assert higher_is_better is True


def test_pearson_early_stop_disables_builtin_l2_metric() -> None:
    if trainer_module.lgb is None:
        pytest.skip("lightgbm is not installed")
    train_day_one = pd.Timestamp("2026-01-05 14:30")
    train_day_two = pd.Timestamp("2026-01-06 14:30")
    val_day = pd.Timestamp("2026-01-07 14:30")
    train = SplitData(
        x=pd.DataFrame({"f1": range(10), "f2": list(reversed(range(10)))}),
        y=pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.02, -0.01, 0.01, 0.04, -0.03]),
        dt=pd.Series([train_day_one] * 5 + [train_day_two] * 5),
        code=pd.Series([f"c{i}" for i in range(10)]),
    )
    val = SplitData(
        x=pd.DataFrame({"f1": range(5), "f2": list(reversed(range(5)))}),
        y=pd.Series([0.02, -0.01, 0.03, -0.02, 0.01]),
        dt=pd.Series([val_day] * 5),
        code=pd.Series([f"v{i}" for i in range(5)]),
    )

    _, meta = fit_lgbm(
        train=train,
        val=val,
        lgbm_params={
            "objective": "regression",
            "verbosity": -1,
            "device": "cpu",
            "n_estimators": 8,
            "learning_rate": 0.1,
            "num_leaves": 2,
            "min_data_in_leaf": 1,
            "random_state": 42,
        },
        early_stopping_rounds=2,
        label_target_transform=LabelTargetTransformSpec(
            mode="zscore_day",
            loss_day_mass="equal",
        ),
        early_stopping_metric="pearson_ic",
    )

    assert meta["early_stopping_metric"] == "pearson_ic"
    assert meta["metric"] == "None"
    assert meta["label_target_transform"]["train"]["loss_day_mass_applied"] is True
