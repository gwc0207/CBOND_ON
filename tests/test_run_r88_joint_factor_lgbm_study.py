from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import harness.tools.run_r88_joint_factor_lgbm_study as study
from cbond_on.infra.factors.canonical_store import CanonicalFactorStore, FactorColumnContract, FactorTableContract
from cbond_on.infra.factors.r88_experiment_table import R88ExperimentTableError, admit_r88_experiment_table


_RUNTIME_GLOBALS = (
    "STUDY_ID", "DEFAULT_STUDY_ROOT", "FACTOR_ROOT", "BACKFILL_MANIFEST", "R88_INPUT_MODE",
    "HOLDOUT_END", "DEFAULT_END", "SCRATCH_PARENT", "PROFILE_PATH", "CANONICAL_FACTOR_STORE_ROOT",
    "EXPERIMENT_TABLE_ROOT", "EXPERIMENT_TABLE_MANIFEST", "_R88_BINDING",
)


@pytest.fixture(autouse=True)
def _restore_runtime_globals() -> None:
    before = {name: getattr(study, name) for name in _RUNTIME_GLOBALS}
    try:
        yield
    finally:
        for name, value in before.items():
            setattr(study, name, value)


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _profile(path: Path) -> tuple[dict[str, object], list[str]]:
    factors = [f"factor_{index:03d}" for index in range(88)]
    specs = [
        {"name": factor, "factor": "test_factor", "params": {}, "output_col": None, "rust_contract_id": f"test/{factor}"}
        for factor in factors
    ]
    payload: dict[str, object] = {
        "schema_version": "research_r88_factor_profile/v1",
        "admission_profile": "r88_test",
        "factors": factors,
        "factor_specs": specs,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, factors


def _mark_ready(store: CanonicalFactorStore, table_id: str) -> None:
    coverage = {"test": "full", "table_id": table_id}
    migration_id = f"test-{table_id}"
    store.record_migration_attestation(table_id, {"migration_id": migration_id, "scope": "full", "expected_coverage": coverage})
    store.mark_migration_verified(table_id, migration_id=migration_id, coverage=coverage)


def _canonical_r88_table(
    tmp_path: Path,
    *,
    direct_current: bool = False,
) -> tuple[Path, Path, dict[str, object], list[str], date]:
    profile_path = tmp_path / "profile.json5"
    profile, factor_ids = _profile(profile_path)
    store = CanonicalFactorStore(tmp_path / "factor_store")
    store.initialize()
    contract = FactorTableContract(
        tuple(
            FactorColumnContract(
                factor_id=factor_id,
                factor_version="v1",
                contract_hash=hashlib.sha256(factor_id.encode("utf-8")).hexdigest(),
                output_column=factor_id,
            )
            for factor_id in factor_ids
        )
    )
    day = date(2026, 7, 30)
    timestamp = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    frame = pd.DataFrame(
        {factor_id: [float(index)] for index, factor_id in enumerate(factor_ids)},
        index=pd.MultiIndex.from_tuples([(timestamp, "110001.SH")], names=["dt", "code"]),
    )
    profile_evidence = {
        "path": profile_path.resolve().as_posix(),
        "admission_profile": profile["admission_profile"],
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "specs_sha256": _hash_json(profile["factor_specs"]),
    }
    source_evidence = (
        {"source": "r88_full_current_rust88_recompute", "profile": profile_evidence}
        if direct_current
        else {
            "fragments": [{
                "source": "experiment_input",
                "historical_contract_evidence": {
                    "historical_profile": profile_evidence,
                    "source_execution_status": "completed_rust88_backfill",
                    "source_model_training_ready": True,
                },
            }]
        }
    )
    store.write_day("experiment", day, frame, contract=contract, source_evidence=source_evidence)
    _mark_ready(store, "experiment")
    _mark_ready(store, "live")
    return store.root, profile_path, profile, factor_ids, day


def test_r88_admission_reads_only_manifest_bound_canonical_experiment_table(tmp_path: Path) -> None:
    root, profile_path, profile, factor_ids, day = _canonical_r88_table(tmp_path)
    binding = admit_r88_experiment_table(
        {"factor_table": {"table_id": "experiment", "root": str(root)}},
        profile_path=profile_path,
        profile=profile,
    )
    assert binding.table_root == root / "experiment"
    assert binding.days == (day,)
    assert binding.contract.factor_ids == tuple(factor_ids)
    assert binding.reader.read_day(day).columns.tolist() == factor_ids
    assert binding.to_evidence()["table_id"] == "experiment"


def test_r88_admission_accepts_a_direct_current_catalog_publish_with_the_same_profile_contract(tmp_path: Path) -> None:
    root, profile_path, profile, _factor_ids, day = _canonical_r88_table(tmp_path, direct_current=True)
    binding = admit_r88_experiment_table(
        {"factor_table": {"table_id": "experiment", "root": str(root)}},
        profile_path=profile_path,
        profile=profile,
    )
    assert binding.days == (day,)
    assert binding.source_provenance["materialization"] == "current_catalog_direct"


def test_r88_admission_rejects_direct_root_or_noncanonical_table(tmp_path: Path) -> None:
    root, profile_path, profile, _factor_ids, _day = _canonical_r88_table(tmp_path)
    with pytest.raises(R88ExperimentTableError, match="explicit canonical factor_table"):
        admit_r88_experiment_table({}, profile_path=profile_path, profile=profile)
    with pytest.raises(R88ExperimentTableError, match="table_id='experiment'"):
        admit_r88_experiment_table(
            {"factor_table": {"table_id": "live", "root": str(root)}}, profile_path=profile_path, profile=profile
        )


def test_r88_admission_rejects_profile_formula_contract_drift(tmp_path: Path) -> None:
    root, profile_path, profile, _factor_ids, _day = _canonical_r88_table(tmp_path)
    changed = dict(profile)
    changed_specs = [dict(item) for item in profile["factor_specs"]]
    changed_specs[0]["params"] = {"changed": True}
    changed["factor_specs"] = changed_specs
    with pytest.raises(R88ExperimentTableError, match="factor-spec hash differs"):
        admit_r88_experiment_table(
            {"factor_table": {"table_id": "experiment", "root": str(root)}}, profile_path=profile_path, profile=changed
        )


def test_study_runtime_rejects_legacy_factor_data_cli_override() -> None:
    with pytest.raises(ValueError, match="migration/audit provenance only"):
        study._configure_runtime(
            factor_root_text="D:/cbond_on/research_scratch/old/factor_data", manifest_text=None, study_id_text=None
        )


def test_study_runtime_generates_canonical_experiment_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, profile_path, _profile_payload, factor_ids, day = _canonical_r88_table(tmp_path)
    monkeypatch.setattr(study, "CANONICAL_FACTOR_STORE_ROOT", root)
    monkeypatch.setattr(study, "EXPERIMENT_TABLE_ROOT", root / "experiment")
    monkeypatch.setattr(study, "EXPERIMENT_TABLE_MANIFEST", root / "experiment" / "table_manifest.json")
    monkeypatch.setattr(study, "PROFILE_PATH", profile_path)
    configured = study._configure_runtime(factor_root_text=None, manifest_text=None, study_id_text="r88_table_test")
    profile, evidence, observed_factors, days = study._load_r88_contract()
    paths = study._paths_config(tmp_path / "run")
    assert configured.mode == "canonical_experiment"
    assert configured.factor_root == root / "experiment"
    assert observed_factors == factor_ids
    assert days == [day]
    assert evidence["table_id"] == "experiment"
    assert profile["factors"] == factor_ids
    assert paths["factor_table"] == {"table_id": "experiment", "root": str(root)}
    assert "factor_data_root" not in paths["read_only_input_roots"]


def test_study_calendar_requires_exact_factor_label_and_prior_history_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = [date(2024, 12, 27), date(2024, 12, 30)]
    scores = [date(2025, 1, 2), date(2025, 1, 3)]
    label_root = tmp_path / "labels"
    for day in [*history, *scores]:
        path = label_root / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"label")
    monkeypatch.setattr(study, "LABEL_ROOT", label_root)
    monkeypatch.setattr(study, "PRIOR_HISTORY_DAYS", len(history))
    monkeypatch.setattr(study, "list_trading_days_from_raw", lambda *_args, **_kwargs: list(scores))
    monkeypatch.setattr(study, "prev_trading_days_from_raw", lambda *_args, **_kwargs: list(history))

    evidence = study._require_exact_study_calendar(
        factor_days=[*history, *scores], start=scores[0], end=scores[-1]
    )

    assert evidence["score_days"] == [day.isoformat() for day in scores]
    assert evidence["prior_history_days"] == [day.isoformat() for day in history]
    (label_root / f"{scores[-1]:%Y-%m}" / f"{scores[-1]:%Y%m%d}.parquet").unlink()
    with pytest.raises(RuntimeError, match="label coverage is incomplete"):
        study._require_exact_study_calendar(
            factor_days=[*history, *scores], start=scores[0], end=scores[-1]
        )


def test_plan_objective_windows_keep_2025_selection_and_extend_reporting_end() -> None:
    development, reporting = study._objective_windows(
        {
            "primary_objective": {
                "development": {"start": "2025-01-02", "end": "2025-12-31"},
                "holdout": {"start": "2026-01-01", "end": "2026-08-27"},
            }
        }
    )
    assert development == (date(2025, 1, 2), date(2025, 12, 31))
    assert reporting == (date(2026, 1, 1), date(2026, 8, 27))


def test_prepare_rejects_any_range_other_than_the_approved_2025_to_2026_chain() -> None:
    with pytest.raises(ValueError, match="fixed approved range"):
        study.prepare(
            Path("not_used_after_date_guard"),
            "r88_invalid_range",
            date(2025, 1, 3),
            study.R88_EXPERIMENT_HOLDOUT_END,
        )
    with pytest.raises(ValueError, match="fixed approved range"):
        study.prepare(
            Path("not_used_after_date_guard"),
            "r88_invalid_range",
            study.DEV_START,
            date(2026, 8, 26),
        )


def test_candidate_matrix_is_exactly_five_fixed_sets_by_six_warm_profiles() -> None:
    profile = study.load_json_like(study.PROFILE_PATH)
    factors = [str(value) for value in profile["factors"]]
    selections = {
        top_k: {"selected_factors": factors[:top_k]}
        for top_k in (25, 35, 50)
    }

    candidates = study._candidate_specs(profile, factors, selections)

    assert len(candidates) == 30
    assert {spec["factor_set_id"] for spec in candidates.values()} == {
        "full88", "r50", "icir25", "icir35", "icir50"
    }
    assert {spec["hyperparameter_id"] for spec in candidates.values()} == set(study._lgbm_profiles())
    assert {spec["state_mode"] for spec in candidates.values()} == {"warm_start"}


def test_r88_warm_candidate_pins_canonical_metadata_and_strict_no_cold_fallback(tmp_path: Path) -> None:
    spec = {
        "state_mode": "warm_start", "factor_set_id": "full88", "hyperparameter_id": "p1", "factor_mode": "fixed_full88",
        "factors": ["factor_a", "factor_b"], "effective_selected_factors": ["factor_a", "factor_b"],
        "lgbm_params": {"feature_fraction": 1.0, "feature_fraction_bynode": 1.0}, "description": "test",
    }
    cfg = study._model_config("r88_test", spec, tmp_path / "run", date(2025, 1, 2), date(2026, 7, 30))
    assert cfg["incremental"]["warm_start"] is True
    assert cfg["incremental"]["save_state"] is True
    assert cfg["incremental"]["skip_existing_scores"] is False
    assert cfg["incremental"]["strict_warm_start"] == {"enabled": True, "require_initial_checkpoint": False}
    assert cfg["research_only"]["factor_table"]["table_id"] == "experiment"
    assert "factor_store_root" not in cfg["research_only"]


def test_strict_warm_start_audit_rejects_a_silent_cold_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    days = [date(2025, 1, 2), date(2025, 1, 3)]
    monkeypatch.setattr(study, "_load_r88_contract", lambda: ({}, {}, [], days))
    run_root = tmp_path / "run"
    candidate = "r88_test"
    state_dir = run_root / "runtime" / "results" / "model_state" / candidate
    state_dir.mkdir(parents=True)
    for day in days:
        (state_dir / f"{day:%Y-%m-%d}.txt").write_text("checkpoint", encoding="utf-8")
    model_config = run_root / "configs" / "model.json5"
    study._write_json(model_config, {"incremental": {"state_dir": str(state_dir), "strict_warm_start": {"enabled": True, "require_initial_checkpoint": False}}})
    rolling_path = run_root / "runtime" / "results" / "models" / candidate / "test" / "rolling_metrics.csv"
    rolling_path.parent.mkdir(parents=True)
    pd.DataFrame([
        {"trade_date": days[0], "strict_warm_start": True, "warm_start_active": False, "warm_start_required": False, "warm_start_bootstrap": True, "warm_start_checkpoint": "", "warm_start_source_day": ""},
        {"trade_date": days[1], "strict_warm_start": True, "warm_start_active": False, "warm_start_required": False, "warm_start_bootstrap": False, "warm_start_checkpoint": "", "warm_start_source_day": ""},
    ]).to_csv(rolling_path, index=False)
    row = {"candidate": candidate, "state_mode": "warm_start", "model_config": str(model_config), "start": str(days[0]), "end": str(days[-1])}
    with pytest.raises(RuntimeError, match="strict warm-start audit failed"):
        study._audit_strict_warm_start(run_root, row, candidate)
    audit = json.loads((run_root / "warm_start_coverage" / f"{candidate}.json").read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert any("strict warm-start flags" in item for item in audit["errors"])


def test_strict_warm_start_audit_accepts_continuation_across_year_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = [date(2025, 12, 31), date(2026, 1, 5)]
    monkeypatch.setattr(study, "_load_r88_contract", lambda: ({}, {}, [], days))
    run_root = tmp_path / "run"
    candidate = "r88_test"
    state_dir = run_root / "runtime" / "results" / "model_state" / candidate
    state_dir.mkdir(parents=True)
    for day in days:
        (state_dir / f"{day:%Y-%m-%d}.txt").write_text("checkpoint", encoding="utf-8")
    model_config = run_root / "configs" / "model.json5"
    study._write_json(
        model_config,
        {"incremental": {"state_dir": str(state_dir), "strict_warm_start": {"enabled": True, "require_initial_checkpoint": False}}},
    )
    rolling_path = run_root / "runtime" / "results" / "models" / candidate / "test" / "rolling_metrics.csv"
    rolling_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": days[0], "strict_warm_start": True, "warm_start_active": False, "warm_start_required": False, "warm_start_bootstrap": True, "warm_start_checkpoint": "", "warm_start_source_day": ""},
            {"trade_date": days[1], "strict_warm_start": True, "warm_start_active": True, "warm_start_required": True, "warm_start_bootstrap": False, "warm_start_checkpoint": "2025-12-31.txt", "warm_start_source_day": "2025-12-31"},
        ]
    ).to_csv(rolling_path, index=False)
    row = {"candidate": candidate, "state_mode": "warm_start", "model_config": str(model_config), "start": str(days[0]), "end": str(days[-1])}

    audit = study._audit_strict_warm_start(run_root, row, candidate)

    assert audit["status"] == "ok"
    assert audit["actual_checkpoint_names"] == ["2025-12-31.txt", "2026-01-05.txt"]
