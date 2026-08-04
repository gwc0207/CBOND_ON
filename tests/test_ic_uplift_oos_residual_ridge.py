from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import harness.tools.ic_uplift_oos_residual_ridge as residual_tool

from harness.tools.ic_uplift_oos_residual_ridge import (
    PRIOR_INTRADAY_SHARPE_120D_FEATURE,
    PRIOR_INTRADAY_RETURN_SURPRISE_120D_FEATURE,
    PARITY_V2_FEATURE,
    WAVE80_LIQ_VOL_BALANCE_45M_L5_FEATURE,
    CandidatePlan,
    _build_calendar_fallback_code_audit,
    _build_calendar_fallback_inputs,
    _feature_columns_for_plans,
    _fit_calendar_fallback_candidate,
    _plans_for_set,
    _read_factor_frame,
    training_days,
)


def test_training_days_excludes_the_score_day() -> None:
    days = ["2024-05-08", "2024-05-09", "2024-05-10", "2024-05-13"]

    assert training_days(days, 3, 120) == ["2024-05-08", "2024-05-09", "2024-05-10"]
    assert "2024-05-13" not in training_days(days, 3, 120)


def test_training_days_honours_lookback_without_using_current_day() -> None:
    days = ["2024-05-08", "2024-05-09", "2024-05-10", "2024-05-13"]

    assert training_days(days, 3, 2) == ["2024-05-09", "2024-05-10"]
    assert "2024-05-13" not in training_days(days, 3, 2)


def test_parity_candidate_set_is_frozen_and_uses_the_sidecar_feature() -> None:
    plans = _plans_for_set("parity_v2_r1")

    assert [plan.target_mode for plan in plans] == ["anchored_residual", "anchored_residual"]
    assert _feature_columns_for_plans(plans) == (PARITY_V2_FEATURE, "vwap_30m")


def test_tail_path_candidate_is_a_single_fixed_sidecar_fallback_arm() -> None:
    plans = _plans_for_set("tail_path_efficiency_5m_fallback_r1")

    assert len(plans) == 1
    plan = plans[0]
    assert plan.feature_columns == ("tail_path_efficiency_5m_v1",)
    assert plan.target_mode == "anchored_residual"
    assert plan.lookback_days == 120
    assert plan.alpha == 20.0
    assert plan.availability_policy == "calendar_regsim_fallback"
    assert plan.min_train_feature_days == 96
    assert plan.min_score_coverage == 0.80


def test_parity_candidate_reads_only_the_explicit_sidecar_root(tmp_path) -> None:
    day = "2026-01-05"
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-01-05 14:30:00"), "110001.SH"), (pd.Timestamp("2026-01-05 14:30:00"), "110002.SH")],
        names=["dt", "code"],
    )
    primary_path = tmp_path / "primary" / "2026-01" / "20260105.parquet"
    sidecar_path = tmp_path / "sidecar" / "2026-01" / "20260105.parquet"
    primary_path.parent.mkdir(parents=True)
    sidecar_path.parent.mkdir(parents=True)
    pd.DataFrame({"vwap_30m": [100.0, 101.0]}, index=index).to_parquet(primary_path)
    pd.DataFrame({PARITY_V2_FEATURE: [0.01, -0.02]}, index=index).to_parquet(sidecar_path)

    actual = _read_factor_frame(
        primary_path.parents[1],
        sidecar_path.parents[1],
        day,
        (PARITY_V2_FEATURE, "vwap_30m"),
    )

    assert actual is not None
    assert actual["code"].tolist() == ["110001.SH", "110002.SH"]
    assert actual[f"{PARITY_V2_FEATURE}__z"].notna().all()
    assert actual["vwap_30m__z"].notna().all()


def test_wave80_candidate_is_frozen_and_reads_the_primary_store(tmp_path) -> None:
    plans = _plans_for_set("wave80_r1")

    assert [plan.target_mode for plan in plans] == ["anchored_residual", "anchored_residual"]
    assert _feature_columns_for_plans(plans) == (
        WAVE80_LIQ_VOL_BALANCE_45M_L5_FEATURE,
        "vwap_30m",
    )

    day = "2026-01-05"
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-05 14:30:00"), "110001.SH"),
            (pd.Timestamp("2026-01-05 14:30:00"), "110002.SH"),
        ],
        names=["dt", "code"],
    )
    primary_path = tmp_path / "primary" / "2026-01" / "20260105.parquet"
    primary_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            WAVE80_LIQ_VOL_BALANCE_45M_L5_FEATURE: [0.01, -0.02],
            "vwap_30m": [100.0, 101.0],
        },
        index=index,
    ).to_parquet(primary_path)

    actual = _read_factor_frame(
        primary_path.parents[1],
        None,
        day,
        _feature_columns_for_plans(plans),
    )

    assert actual is not None
    assert actual["code"].tolist() == ["110001.SH", "110002.SH"]
    assert actual[f"{WAVE80_LIQ_VOL_BALANCE_45M_L5_FEATURE}__z"].notna().all()


def _synthetic_sections_and_labels(days: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    codes = [f"110{i:03d}.SH" for i in range(30)]
    values = np.linspace(-1.0, 1.0, len(codes))
    sections = {}
    labels = {}
    for day in days:
        sections[day] = pd.DataFrame(
            {
                "score_day": day,
                "code": codes,
                "regsim_score": values,
                "regsim_score__z": values,
                f"{PARITY_V2_FEATURE}__z": values * 0.5,
                "vwap_30m__z": values * -0.25,
            }
        )
        labels[day] = pd.DataFrame({"code": codes, "y": values, "y__z": values})
    return sections, labels


def test_score_stage_keeps_regsim_baseline_for_paired_evaluation(monkeypatch) -> None:
    days = pd.bdate_range("2024-01-02", periods=121).strftime("%Y-%m-%d").tolist()
    sections, labels = _synthetic_sections_and_labels(days)

    monkeypatch.setattr(residual_tool, "_read_label", lambda _root, day: labels[day].copy())
    plans = _plans_for_set("parity_v2_r1")
    scores, audit, _ = residual_tool._fit_daily_candidates(
        sections,
        Path("unused"),
        plans=plans,
        feature_columns=_feature_columns_for_plans(plans),
        score_calendar=days,
    )

    assert "regsim" in scores.columns
    assert set(scores.columns) >= {"regsim", *(plan.name for plan in plans)}
    assert set(scores["score_day"]) == {days[120]}
    assert audit["train_days_used"].tolist() == [120, 120]
    assert audit["train_days_strict_calendar_window"].all()


def test_strict_calendar_window_does_not_bridge_a_missing_factor_day(monkeypatch) -> None:
    days = pd.bdate_range("2024-01-02", periods=242).strftime("%Y-%m-%d").tolist()
    sections, labels = _synthetic_sections_and_labels(days)
    missing_day = days[60]
    del sections[missing_day]
    monkeypatch.setattr(residual_tool, "_read_label", lambda _root, day: labels[day].copy())

    plans = _plans_for_set("parity_v2_r1")
    scores, _, metadata = residual_tool._fit_daily_candidates(
        sections,
        Path("unused"),
        plans=plans,
        feature_columns=_feature_columns_for_plans(plans),
        score_calendar=days,
    )

    assert scores["score_day"].min() == days[181]
    assert days[120] in metadata["score_stage_skipped"]
    assert "missing score/factor section" in metadata["score_stage_skipped"][days[120]]


def test_score_for_day_is_independent_of_its_same_day_label(monkeypatch) -> None:
    days = pd.bdate_range("2024-01-02", periods=121).strftime("%Y-%m-%d").tolist()
    sections, labels = _synthetic_sections_and_labels(days)
    plans = _plans_for_set("parity_v2_r1")

    monkeypatch.setattr(residual_tool, "_read_label", lambda _root, day: labels[day].copy())
    before, _, _ = residual_tool._fit_daily_candidates(
        sections,
        Path("unused"),
        plans=plans,
        feature_columns=_feature_columns_for_plans(plans),
        score_calendar=days,
    )
    labels[days[120]]["y"] *= -1.0
    labels[days[120]]["y__z"] *= -1.0
    after, _, _ = residual_tool._fit_daily_candidates(
        sections,
        Path("unused"),
        plans=plans,
        feature_columns=_feature_columns_for_plans(plans),
        score_calendar=days,
    )

    pd.testing.assert_frame_equal(before, after)


def test_evaluate_refuses_to_open_labels_without_completed_score_artifacts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        residual_tool,
        "_read_label",
        lambda *_args, **_kwargs: pytest.fail("evaluate must not read labels before score artifacts exist"),
    )

    with pytest.raises(FileNotFoundError):
        residual_tool.evaluate(
            Namespace(
                output_root=str(tmp_path),
                label_root="unused",
                validation_start="2025-10-08",
                final_start="2026-05-06",
            )
        )


def _fallback_plan(*, min_train_feature_days: int = 96, min_score_coverage: float = 0.80) -> CandidatePlan:
    return CandidatePlan(
        "synthetic_calendar_fallback",
        (PRIOR_INTRADAY_SHARPE_120D_FEATURE,),
        "anchored_residual",
        120,
        20.0,
        availability_policy="calendar_regsim_fallback",
        min_train_feature_days=min_train_feature_days,
        min_score_coverage=min_score_coverage,
    )


def _fallback_sections_and_labels(
    days: list[str],
    *,
    feature_days: set[str],
    codes: list[str] | None = None,
    feature_codes_by_day: dict[str, list[str]] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    codes = codes or [f"120{i:03d}.SH" for i in range(40)]
    values = np.linspace(-1.0, 1.0, len(codes))
    feature_values = np.sin(np.linspace(-2.0, 2.0, len(codes)))
    code_to_feature = dict(zip(codes, feature_values, strict=True))
    base_sections: dict[str, pd.DataFrame] = {}
    feature_sections: dict[str, pd.DataFrame] = {}
    labels: dict[str, pd.DataFrame] = {}
    for day in days:
        base_sections[day] = pd.DataFrame(
            {
                "code": codes,
                "regsim_score": values,
                "regsim_score__z": values,
            }
        )
        labels[day] = pd.DataFrame(
            {
                "code": codes,
                "y": values + 0.40 * feature_values,
                "y__z": values + 0.40 * feature_values,
            }
        )
        if day in feature_days:
            feature_codes = (feature_codes_by_day or {}).get(day, codes)
            positions = [codes.index(code) for code in feature_codes]
            feature_sections[day] = pd.DataFrame(
                {
                    "score_day": day,
                    "code": feature_codes,
                    "regsim_score": values[positions],
                    "regsim_score__z": values[positions],
                    f"{PRIOR_INTRADAY_SHARPE_120D_FEATURE}__z": [
                        code_to_feature[code] for code in feature_codes
                    ],
                }
            )
    return base_sections, feature_sections, labels


def test_prior_intraday_fallback_plan_is_frozen() -> None:
    plans = _plans_for_set("prior_intraday_sharpe_120d_fallback_r1")

    assert len(plans) == 1
    plan = plans[0]
    assert plan.feature_columns == (PRIOR_INTRADAY_SHARPE_120D_FEATURE,)
    assert plan.target_mode == "anchored_residual"
    assert plan.lookback_days == 120
    assert plan.alpha == 20.0
    assert plan.availability_policy == "calendar_regsim_fallback"
    assert plan.min_train_feature_days == 96
    assert plan.min_score_coverage == 0.80


def test_prior_intraday_return_surprise_fallback_plan_is_frozen() -> None:
    plans = _plans_for_set("prior_intraday_return_surprise_120d_fallback_r1")

    assert len(plans) == 1
    plan = plans[0]
    assert plan.feature_columns == (PRIOR_INTRADAY_RETURN_SURPRISE_120D_FEATURE,)
    assert plan.target_mode == "anchored_residual"
    assert plan.lookback_days == 120
    assert plan.alpha == 20.0
    assert plan.availability_policy == "calendar_regsim_fallback"
    assert plan.min_train_feature_days == 96
    assert plan.min_score_coverage == 0.80


def test_calendar_fallback_inputs_enforce_exact_coverage_threshold(monkeypatch) -> None:
    days = ["2025-01-02", "2025-01-03"]
    codes = [f"120{i:03d}.SH" for i in range(40)]
    plan = _fallback_plan()
    base = pd.DataFrame(
        {
            "code": codes,
            "regsim_score": np.linspace(-1.0, 1.0, 40),
            "regsim_score__z": np.linspace(-1.0, 1.0, 40),
        }
    )

    def fake_factor(_root, _sidecar, day, _columns):
        count = 32 if day == days[0] else 31
        value = np.linspace(-1.0, 1.0, count)
        return pd.DataFrame(
            {
                "code": codes[:count],
                PRIOR_INTRADAY_SHARPE_120D_FEATURE: value,
                f"{PRIOR_INTRADAY_SHARPE_120D_FEATURE}__z": value,
            }
        )

    monkeypatch.setattr(residual_tool, "_read_regsim_score", lambda *_args: base.copy())
    monkeypatch.setattr(residual_tool, "_read_factor_frame", fake_factor)
    base_sections, feature_sections, skipped, availability = _build_calendar_fallback_inputs(
        days,
        Path("primary"),
        Path("sidecar"),
        Path("scores"),
        plan=plan,
    )

    assert set(base_sections) == set(days)
    assert set(feature_sections) == {days[0]}
    assert "below frozen minimum" in skipped[days[1]]
    assert availability.set_index("score_day").loc[days[0], "feature_coverage"] == pytest.approx(0.80)
    assert bool(availability.set_index("score_day").loc[days[0], "feature_usable"])
    assert not bool(availability.set_index("score_day").loc[days[1], "feature_usable"])


@pytest.mark.parametrize("missing_training_days, expected_fitted", [(24, True), (25, False)])
def test_calendar_fallback_uses_only_fixed_120_slots_and_honours_96_day_boundary(
    monkeypatch, missing_training_days, expected_fitted
) -> None:
    days = pd.bdate_range("2024-01-02", periods=121).strftime("%Y-%m-%d").tolist()
    feature_days = set(days[missing_training_days:])
    base_sections, feature_sections, labels = _fallback_sections_and_labels(
        days,
        feature_days=feature_days,
    )
    monkeypatch.setattr(residual_tool, "_read_label", lambda _root, day: labels[day].copy())

    plan = _fallback_plan()
    scores, audit, _ = _fit_calendar_fallback_candidate(
        base_sections,
        feature_sections,
        Path("unused"),
        plan=plan,
        score_calendar=days,
    )

    final_audit = audit.set_index("score_day").loc[days[-1]]
    final_scores = scores.loc[scores["score_day"] == days[-1]]
    assert int(final_audit["train_days_used"]) == 120 - missing_training_days
    assert int(final_audit["train_feature_days_missing"]) == missing_training_days
    assert bool(final_audit["train_calendar_slots_exact"])
    assert bool(final_audit["train_feature_days_complete"]) is False
    assert final_audit["train_feature_max_day"] == days[-2]
    assert bool(final_audit["train_label_max_is_strictly_before_score_day"])
    assert bool(final_audit["fitted"]) is expected_fitted
    if expected_fitted:
        assert not np.array_equal(
            final_scores[plan.name].to_numpy(dtype=float),
            final_scores["regsim"].to_numpy(dtype=float),
        )
    else:
        np.testing.assert_array_equal(
            final_scores[plan.name].to_numpy(dtype=float),
            final_scores["regsim"].to_numpy(dtype=float),
        )
        assert "only 95 usable factor/label days" in final_audit["fallback_reason"]


def test_calendar_fallback_preserves_all_codes_and_only_falls_back_for_missing_current_factor(
    monkeypatch,
) -> None:
    days = pd.bdate_range("2024-01-02", periods=121).strftime("%Y-%m-%d").tolist()
    codes = [f"120{i:03d}.SH" for i in range(40)]
    partial_codes = codes[:32]
    base_sections, feature_sections, labels = _fallback_sections_and_labels(
        days,
        feature_days=set(days),
        codes=codes,
        feature_codes_by_day={days[-1]: partial_codes},
    )
    monkeypatch.setattr(residual_tool, "_read_label", lambda _root, day: labels[day].copy())
    plan = _fallback_plan()
    scores, audit, _ = _fit_calendar_fallback_candidate(
        base_sections,
        feature_sections,
        Path("unused"),
        plan=plan,
        score_calendar=days,
    )
    final_scores = scores.loc[scores["score_day"] == days[-1]].set_index("code")
    final_audit = audit.set_index("score_day").loc[days[-1]]
    assert bool(final_audit["fitted"])
    assert final_scores.index.tolist() == codes
    np.testing.assert_array_equal(
        final_scores.loc[codes[32:], plan.name].to_numpy(dtype=float),
        final_scores.loc[codes[32:], "regsim"].to_numpy(dtype=float),
    )
    assert not np.array_equal(
        final_scores.loc[partial_codes, plan.name].to_numpy(dtype=float),
        final_scores.loc[partial_codes, "regsim"].to_numpy(dtype=float),
    )

    code_audit = _build_calendar_fallback_code_audit(scores, audit, feature_sections, plan=plan)
    final_code_audit = code_audit.loc[code_audit["score_day"] == days[-1]].set_index("code")
    assert not final_code_audit.loc[partial_codes, "used_regsim_fallback"].any()
    assert final_code_audit.loc[codes[32:], "used_regsim_fallback"].all()
    assert set(final_code_audit.loc[codes[32:], "fallback_reason"]) == {
        "code has no usable candidate factor"
    }


def test_calendar_fallback_does_not_read_or_use_its_same_day_label(monkeypatch) -> None:
    days = pd.bdate_range("2024-01-02", periods=121).strftime("%Y-%m-%d").tolist()
    base_sections, feature_sections, labels = _fallback_sections_and_labels(days, feature_days=set(days))
    plan = _fallback_plan()
    monkeypatch.setattr(residual_tool, "_read_label", lambda _root, day: labels[day].copy())
    before, _, _ = _fit_calendar_fallback_candidate(
        base_sections,
        feature_sections,
        Path("unused"),
        plan=plan,
        score_calendar=days,
    )
    labels[days[-1]]["y"] *= -1.0
    labels[days[-1]]["y__z"] *= -1.0
    after, _, _ = _fit_calendar_fallback_candidate(
        base_sections,
        feature_sections,
        Path("unused"),
        plan=plan,
        score_calendar=days,
    )

    pd.testing.assert_frame_equal(before, after)


def test_score_dispatches_calendar_fallback_and_writes_full_universe_audits(monkeypatch, tmp_path) -> None:
    days = pd.bdate_range("2024-01-02", periods=4).strftime("%Y-%m-%d").tolist()
    base_sections, _, _ = _fallback_sections_and_labels(days, feature_days=set())
    availability = pd.DataFrame(
        {
            "score_day": days,
            "regsim_codes": [len(next(iter(base_sections.values())))] * len(days),
            "feature_codes": [0] * len(days),
            "feature_coverage": [0.0] * len(days),
            "feature_usable": [False] * len(days),
            "feature_status": ["missing candidate factor columns or unusable factor frame"] * len(days),
        }
    )
    monkeypatch.setattr(residual_tool, "discover_score_days", lambda *_args: days)
    monkeypatch.setattr(
        residual_tool,
        "_build_calendar_fallback_inputs",
        lambda *_args, **_kwargs: (base_sections, {}, {day: "missing factor" for day in days}, availability.copy()),
    )
    monkeypatch.setattr(
        residual_tool,
        "_fit_daily_candidates",
        lambda *_args, **_kwargs: pytest.fail("strict scorer must not run for calendar fallback"),
    )

    output_root = tmp_path / "calendar_fallback_output"
    output = residual_tool.score(
        Namespace(
            output_root=str(output_root),
            factor_root=str(tmp_path / "primary"),
            sidecar_factor_root=str(tmp_path / "sidecar"),
            score_root=str(tmp_path / "scores"),
            label_root=str(tmp_path / "labels"),
            start=days[0],
            end=days[-1],
            candidate_set="prior_intraday_sharpe_120d_fallback_r1",
        )
    )

    assert output == output_root
    assert (output_root / "oof_scores.parquet").is_file()
    assert (output_root / "availability_audit.csv").is_file()
    assert (output_root / "fallback_code_audit.csv").is_file()
    manifest = json.loads((output_root / "score_manifest.json").read_text(encoding="utf-8"))
    assert manifest["availability_policy"] == ["calendar_regsim_fallback"]
    assert manifest["oof_matches_base_regsim_universe"] is True
    assert manifest["fallback_code_count"] == 4 * 40
    scored = pd.read_parquet(output_root / "oof_scores.parquet")
    candidate = _plans_for_set("prior_intraday_sharpe_120d_fallback_r1")[0].name
    np.testing.assert_array_equal(scored[candidate].to_numpy(dtype=float), scored["regsim"].to_numpy(dtype=float))


def test_score_rejects_mixed_strict_and_calendar_fallback_policies(monkeypatch, tmp_path) -> None:
    strict_plan = CandidatePlan("strict", ("vwap_30m",), "anchored_residual", 120, 20.0)
    fallback_plan = _fallback_plan()
    monkeypatch.setattr(residual_tool, "_plans_for_set", lambda _candidate_set: (strict_plan, fallback_plan))
    monkeypatch.setattr(residual_tool, "discover_score_days", lambda *_args: ["2025-01-02"])

    with pytest.raises(ValueError, match="cannot mix strict and calendar_regsim_fallback"):
        residual_tool.score(
            Namespace(
                output_root=str(tmp_path / "mixed"),
                factor_root=str(tmp_path / "primary"),
                sidecar_factor_root=str(tmp_path / "sidecar"),
                score_root=str(tmp_path / "scores"),
                label_root=str(tmp_path / "labels"),
                start="2025-01-02",
                end="2025-01-02",
                candidate_set="synthetic",
            )
        )


def test_evaluate_calendar_fallback_keeps_candidate_and_regsim_n_aligned(monkeypatch, tmp_path) -> None:
    days = ["2025-10-08", "2026-05-06"]
    codes = [f"120{i:03d}.SH" for i in range(40)]
    values = np.linspace(-1.0, 1.0, len(codes))
    candidate = "synthetic_calendar_fallback"
    output_root = tmp_path / "evaluate_calendar_fallback"
    output_root.mkdir()
    scores = pd.concat(
        [
            pd.DataFrame(
                {
                    "score_day": day,
                    "code": codes,
                    "regsim": values,
                    candidate: values,
                }
            )
            for day in days
        ],
        ignore_index=True,
    )
    scores.to_parquet(output_root / "oof_scores.parquet", index=False)
    (output_root / "score_manifest.json").write_text(
        json.dumps(
            {
                "availability_policy": ["calendar_regsim_fallback"],
                "oof_matches_base_regsim_universe": True,
                "fallback_score_day_count": len(days),
                "fallback_code_count": len(scores),
                "fallback_code_share": 1.0,
                "availability_audit_path": "availability_audit.csv",
                "fallback_code_audit_path": "fallback_code_audit.csv",
            }
        ),
        encoding="utf-8",
    )
    labels = {
        day: pd.DataFrame({"code": codes, "y": values[::-1], "y__z": values[::-1]})
        for day in days
    }
    monkeypatch.setattr(residual_tool, "_read_label", lambda _root, day: labels[day].copy())

    residual_tool.evaluate(
        Namespace(
            output_root=str(output_root),
            label_root="unused",
            validation_start="2025-10-08",
            final_start="2026-05-06",
        )
    )

    daily = pd.read_csv(output_root / "daily_metrics.csv")
    n_by_day_prediction = daily.pivot(index="score_day", columns="prediction", values="n")
    np.testing.assert_array_equal(
        n_by_day_prediction["regsim"].to_numpy(dtype=int),
        n_by_day_prediction[candidate].to_numpy(dtype=int),
    )
    evaluation_manifest = json.loads((output_root / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert evaluation_manifest["availability_policy"] == ["calendar_regsim_fallback"]
    assert evaluation_manifest["score_universe_integrity"]["oof_matches_base_regsim_universe"] is True
    assert evaluation_manifest["universe_scope"].startswith("full Regsim score universe")
