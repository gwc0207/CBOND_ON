from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from cbond_on.workflows.research import factor_supplement as supplement


def _write_catalog(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "factor"])
        writer.writeheader()
        for family, factor in rows:
            writer.writerow({"family": family, "factor": factor})


def _write_legacy_release(path: Path, factors: list[str]) -> None:
    path.write_text(json.dumps({"factors": factors}), encoding="utf-8")


def _ready_datahub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean = tmp_path / "clean.json"
    clean.write_text(
        json.dumps(
            {
                "required_profile": "cbond_on_live_t1430",
                "validation": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        supplement,
        "run_publish_status",
        lambda **_: {
            "ready": True,
            "done_exists": True,
            "manifest_run_id_consistent": True,
            "manifest_run_id_complete": True,
            "manifests": {"clean": {"path": str(clean), "status": "success"}},
        },
    )


def _legacy_plan_config(catalog: Path, release: Path) -> dict:
    """Test-only compatibility fixture; production config uses canonical JSON."""

    return {
        "schema": supplement.SUPPLEMENT_SCHEMA,
        "research_only": True,
        "schedule": {"time": "23:59"},
        "compute": {
            "engine": "research_python",
            "execution_policy": "research_catalog_python",
        },
        "catalog": {
            "format": "csv_factor_family_catalog",
            "path": str(catalog),
            "expected_factor_count": 3,
        },
        "release_exclude": {
            "format": "legacy_factors_list_for_test_only",
            "manifest": str(release),
        },
        "data_hub": {
            "manifest_root": str(catalog.parent / "manifests"),
            "require_datasets": ["clean"],
            "required_profile": "cbond_on_live_t1430",
        },
        "output": {"scratch_root": str(catalog.parent / "scratch" / "supplement")},
    }


def test_legacy_inventory_is_plan_only_and_release_is_used_only_as_exclusion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.csv"
    release = tmp_path / "release.json"
    _write_catalog(catalog, [("a", "live_a"), ("b", "research_b"), ("c", "research_c")])
    _write_legacy_release(release, ["live_a", "outside_catalog"])
    _ready_datahub(monkeypatch, tmp_path)

    plan = supplement.build_plan(_legacy_plan_config(catalog, release), score_day=date(2026, 8, 25))

    assert plan["disposition"] == "PLAN_READY_NO_EXECUTION"
    assert plan["release_exclude"]["use"] == "exclude_only"
    assert plan["release_exclude"]["excluded_catalog_factor_ids"] == ["live_a"]
    assert plan["release_exclude"]["release_factor_ids_not_in_catalog"] == ["outside_catalog"]
    assert plan["supplement_scope"]["non_live_catalog_factor_ids"] == ["research_b", "research_c"]
    assert plan["research_python_admission"]["permit_required"] is True

    with pytest.raises(Exception, match="canonical factor_catalog_json"):
        supplement.issue_catalog_execution_permit(_legacy_plan_config(catalog, release), plan=plan)


def test_production_config_resolves_canonical_800_catalog_and_50_instance_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ready_datahub(monkeypatch, tmp_path)
    _path, cfg = supplement.load_supplement_config()

    plan = supplement.build_plan(cfg, score_day=date(2026, 8, 25))

    assert plan["catalog"]["format"] == "factor_catalog_json"
    assert plan["catalog"]["factor_count"] == 800
    assert plan["release_exclude"]["format"] == "live_release_instances"
    assert plan["release_exclude"]["factor_count"] == 50
    assert plan["supplement_scope"]["live_excluded_factor_count"] == 50
    assert plan["supplement_scope"]["non_live_catalog_factor_count"] == 750
    assert plan["disposition"] == "PLAN_READY_NO_EXECUTION"


def test_dry_run_writes_only_an_immutable_research_ledger_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(supplement, "RESEARCH_SCRATCH_PARENT", scratch_parent)
    catalog = tmp_path / "catalog.csv"
    release = tmp_path / "release.json"
    _write_catalog(catalog, [("a", "live_a"), ("b", "research_b"), ("c", "research_c")])
    _write_legacy_release(release, ["live_a"])
    _ready_datahub(monkeypatch, tmp_path)
    cfg = _legacy_plan_config(catalog, release)
    cfg["output"]["scratch_root"] = str(scratch_parent / "factor_supplement_v1")

    plan, ledger_path = supplement.run(cfg, score_day=date(2026, 8, 25), mode="dry-run")

    assert plan["disposition"] == "PLAN_READY_NO_EXECUTION"
    assert ledger_path is not None and ledger_path.is_file()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["research_only"] is True
    assert ledger["plan"]["execution_boundary"]["factor_store_write"] == "canonical_factor_library_only_when_execute"
    assert ledger["plan"]["execution_boundary"]["legacy_scratch_factor_store"] == "forbidden_writer"
    assert not (scratch_parent / "factor_supplement_v1" / "locks" / "2026-08-25.lock").exists()
    assert not list((scratch_parent / "factor_supplement_v1").glob("**/factor_data"))


def test_existing_same_day_lock_fails_closed_without_removal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    root = scratch_parent / "factor_supplement_v1"
    monkeypatch.setattr(supplement, "RESEARCH_SCRATCH_PARENT", scratch_parent)
    catalog = tmp_path / "catalog.csv"
    release = tmp_path / "release.json"
    _write_catalog(catalog, [("a", "live_a"), ("b", "research_b"), ("c", "research_c")])
    _write_legacy_release(release, ["live_a"])
    _ready_datahub(monkeypatch, tmp_path)
    cfg = _legacy_plan_config(catalog, release)
    cfg["output"]["scratch_root"] = str(root)
    lock = root / "locks" / "2026-08-25.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"token": "other"}), encoding="utf-8")

    with pytest.raises(supplement.FactorSupplementLockError):
        supplement.run(cfg, score_day=date(2026, 8, 25), mode="dry-run")

    assert lock.is_file()


def test_datahub_failure_stays_fail_closed_before_any_compute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.csv"
    release = tmp_path / "release.json"
    _write_catalog(catalog, [("a", "live_a"), ("b", "research_b"), ("c", "research_c")])
    _write_legacy_release(release, ["live_a"])
    monkeypatch.setattr(
        supplement,
        "run_publish_status",
        lambda **_: {
            "ready": False,
            "done_exists": False,
            "manifest_run_id_consistent": False,
            "manifest_run_id_complete": False,
            "manifests": {},
        },
    )

    plan = supplement.build_plan(_legacy_plan_config(catalog, release), score_day=date(2026, 8, 25))

    assert plan["datahub"]["passed"] is False
    assert "datahub_publish_gate_not_ready" in plan["blocking_reasons"]
    assert plan["execution_boundary"]["factor_compute"] == "not_started"


def test_compact_summary_derives_completed_execute_state_without_changing_plan_state() -> None:
    plan_summary = supplement.compact_summary(
        {
            "disposition": "PLAN_READY_NO_EXECUTION",
            "research_python_admission": {"permit_required": True, "permit_issued": False},
            "execution": {},
        }
    )
    assert plan_summary["research_python_permit_issued"] is False
    assert plan_summary["factor_compute"] == "not_started"

    completed_summary = supplement.compact_summary(
        {
            "disposition": "EXECUTION_COMPLETED",
            "research_python_admission": {"permit_required": True, "permit_issued": False},
            "execution": {
                "requested": True,
                "started": True,
                "status": "completed",
                "manifest_path": "D:/cbond_on/research_scratch/factor_supplement_v1/manifests/x.json",
            },
        }
    )
    assert completed_summary["research_python_permit_issued"] is True
    assert completed_summary["factor_compute"] == "completed"
    assert completed_summary["execution"]["status"] == "completed"


def test_execute_mode_stops_at_datahub_gate_without_factor_store_or_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(supplement, "RESEARCH_SCRATCH_PARENT", scratch_parent)
    catalog = tmp_path / "catalog.csv"
    release = tmp_path / "release.json"
    _write_catalog(catalog, [("a", "live_a"), ("b", "research_b"), ("c", "research_c")])
    _write_legacy_release(release, ["live_a"])
    cfg = _legacy_plan_config(catalog, release)
    cfg["output"]["scratch_root"] = str(scratch_parent / "factor_supplement_v1")
    monkeypatch.setattr(
        supplement,
        "run_publish_status",
        lambda **_: {
            "ready": False,
            "done_exists": False,
            "manifest_run_id_consistent": False,
            "manifest_run_id_complete": False,
            "manifests": {},
        },
    )
    monkeypatch.setattr(
        supplement,
        "run_factor_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pipeline must not run")),
    )

    plan, ledger_path = supplement.run(cfg, score_day=date(2026, 8, 25), mode="execute")

    assert plan["disposition"] == "BLOCKED_DATAHUB"
    assert plan["execution"]["status"] == "blocked_by_plan"
    assert ledger_path is not None and ledger_path.is_file()
    assert not (scratch_parent / "factor_supplement_v1" / "factor_data").exists()


def test_registration_tool_requires_an_explicit_register_switch_and_exposes_execute_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "harness" / "tools" / "register_factor_supplement_task.ps1").read_text(encoding="utf-8")

    assert "[switch]$Register" in source
    assert "if (-not $Register)" in source
    assert '[ValidateSet("dry-run", "execute")]' in source
    assert "-MultipleInstances IgnoreNew" in source
    assert "-ExecutionTimeLimit (New-TimeSpan -Hours 8)" in source
    assert "-StartWhenAvailable:$false" in source
    assert "StdoutPathPattern" in source
    assert "StderrPathPattern" in source
    assert "run_factor_supplement_task.ps1" in source
    assert "canonical factor-library table" in source
    assert "liveLaunch.scheduler" not in source


def test_workflow_executes_only_through_permit_bound_pipeline_and_never_calls_live_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "cbond_on" / "workflows" / "research" / "factor_supplement.py").read_text(encoding="utf-8")

    assert "research_catalog_execution_permit=permit" in source
    assert "build_permitted_factor_specs(permit)" in source
    assert "publish_factor_library_day" in source
    assert "legacy_scratch_factor_store_write" in source
    assert "datahub_publish_gate_not_ready" in source
    forbidden = (
        "run_factor_build",
        "from cbond_on.app.usecases.live_runtime import",
        "from cbond_on.app.pipelines.live_pipeline import",
        "write_trades_to_db(",
        "run_model_score(",
        "run_once(",
        "liveLaunch.",
    )
    assert not any(token in source for token in forbidden)
