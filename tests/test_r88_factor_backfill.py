from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import harness.tools.r88_factor_backfill as r88
from cbond_on.infra.factors.canonical_store import CanonicalFactorStore, FactorColumnContract, FactorTableContract
from cbond_on.infra.factors.canonical_writer import issue_test_canonical_writer_authority


_R88_FIRST_BATCH = {
    "base_duration_stockvol_interaction",
    "base_stockvol_per_moneyness",
    "rating_current_ordinal",
    "lcc_size_frequency_coupling60",
    "bssrc_upper_rank_tail_alignment60",
    "rlmi_return_trade_size_sign_mutual_information60",
    "base_trigger_progress_ratio",
    "base_trigger_revision_gap",
    "bssrc_lower_rank_tail_alignment60",
    "rdm_joint_reprice_depth_retention",
    "dret_momentum_5",
}
_R88_FIRST_BATCH_NAMESPACE = "research_r88_20260825"


def _canonical_digest(specs: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(
            specs,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _eligible_profile() -> dict[str, object]:
    r88_factor_keys = list(r88.R88_RESEARCH_FACTOR_MODULE_MAP)
    specs = [
        {
            "name": f"factor_{index:03d}",
            "factor": r88_factor_keys[index % len(r88_factor_keys)],
            "params": {"window_minutes": 30, "price_col": "last"},
            "output_col": None,
            "rust_contract_id": (
                f"live50_r5/test_factor_{index:03d}"
                if index < 50
                else f"research_r88_20260825/test_factor_{index:03d}/v1"
            ),
        }
        for index in range(88)
    ]
    return {
        "schema_version": r88._PROFILE_SCHEMA,
        "registry": "../registry.json5",
        "research_only": True,
        "admission_profile": r88._PROFILE_ID,
        "execution_status": r88._ELIGIBLE_STATUS,
        "model_training_ready": False,
        "execution_policy": "rust_first",
        "compute": {"engine": "rust", "execution_policy": "rust_first", "backend": "cpu"},
        "time_contract": {"panel_name": "T1430", "factor_time": "14:30", "label_time": "14:42"},
        "range_contract": {"start": "2024-01-01", "end": "2026-07-30"},
        "factor_count": 88,
        "source_counts": {"legacy27": 27, "screened61": 61},
        "factors": [spec["name"] for spec in specs],
        "factor_specs": specs,
        "research_factor_modules": list(r88.R88_RESEARCH_FACTOR_MODULES),
        "research_kernel_modules": dict(r88.R88_RESEARCH_FACTOR_MODULE_MAP),
        "pending_rust_contract_factors": [],
        "specs_sha256": _canonical_digest(specs),
    }


def _write_profile(path: Path, profile: dict[str, object]) -> Path:
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return path


def _datahub_paths() -> dict[str, str]:
    return {
        "raw_data_root": r88._DATAHUB_RAW_ROOT.as_posix(),
        "clean_data_root": r88._DATAHUB_CLEAN_ROOT.as_posix(),
        "cleaned_data_root": r88._DATAHUB_CLEAN_ROOT.as_posix(),
    }


def _write_published_panel_store(
    root: Path,
    *,
    days: list[date],
    datahub_manifest_root: Path,
) -> None:
    """Create a small committed public-store fixture using the publisher schema."""

    contract = {
        "schema_version": r88._PANEL_STORE_CONTRACT_SCHEMA,
        "panel_name": r88._PANEL_STORE_NAME,
        "logical_panel_time": r88._PANEL_STORE_LOGICAL_TIME,
        "physical_cutoff_time": r88._PANEL_STORE_PHYSICAL_CUTOFF,
        "lead_minutes": 1,
        "panel_mode": "snapshot_sequence",
        "assets": list(r88._PANEL_STORE_ASSETS),
        "count_points": r88._PANEL_STORE_COUNT_POINTS,
        "max_lookback_days": r88._PANEL_STORE_MAX_LOOKBACK_DAYS,
        "schedule_windows": [{"start": "14:30", "end": "14:30"}],
        "snapshot": {},
        "asset_overrides": {},
        "snapshot_columns": None,
    }
    contract_hash = r88._canonical_json_sha256(contract)
    contract_path = r88._panel_store_contract_path(root)
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        json.dumps({**contract, "contract_sha256": contract_hash}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for day in days:
        clean_path = datahub_manifest_root / "clean" / f"{day:%Y-%m-%d}.json"
        done_path = datahub_manifest_root / "publish" / f"{day:%Y-%m-%d}.done"
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        done_path.parent.mkdir(parents=True, exist_ok=True)
        run_id = f"datahub_{day:%Y%m%d}"
        clean_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "trade_day": day.isoformat(),
                    "run_id": run_id,
                    "assets_status": {asset: "success" for asset in r88._PANEL_STORE_ASSETS},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        done_path.write_text(
            json.dumps({"ready": True, "trade_day": day.isoformat(), "run_id": run_id}) + "\n",
            encoding="utf-8",
        )
        assets: dict[str, dict[str, object]] = {}
        for asset in r88._PANEL_STORE_ASSETS:
            path = r88._panel_store_asset_path(root, day, asset)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(
                {"trade_time": [pd.Timestamp(day).replace(hour=14, minute=29)]},
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp(day).replace(hour=14, minute=30), f"{asset}.SH", 0)],
                    names=["dt", "code", "seq"],
                ),
            )
            frame.to_parquet(path)
            assets[asset] = {
                "path": path.resolve().as_posix(),
                "sha256": r88._sha256(path),
                "bytes": path.stat().st_size,
                "rows": 1,
                "codes": 1,
                "logical_dt": pd.Timestamp(day).replace(hour=14, minute=30).isoformat(),
                "physical_cutoff": pd.Timestamp(day).replace(hour=14, minute=29).isoformat(),
                "physical_min": pd.Timestamp(day).replace(hour=14, minute=29).isoformat(),
                "physical_max": pd.Timestamp(day).replace(hour=14, minute=29).isoformat(),
                "index_names": ["dt", "code", "seq"],
            }
        manifest_path = r88._panel_store_manifest_path(root, day)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": r88._PANEL_STORE_MANIFEST_SCHEMA,
            "status": "published",
            "run_id": run_id,
            "trade_day": day.isoformat(),
            "contract_sha256": contract_hash,
            "contract_path": contract_path.resolve().as_posix(),
            "datahub_source": {
                "trade_day": day.isoformat(),
                "datahub_run_id": run_id,
                "clean_manifest": {"path": clean_path.resolve().as_posix(), "sha256": r88._sha256(clean_path)},
                "publish_done": {"path": done_path.resolve().as_posix(), "sha256": r88._sha256(done_path)},
            },
            "assets": assets,
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        commit_path = r88._panel_store_done_path(root, day)
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        commit_path.write_text(
            json.dumps(
                {
                    "schema_version": r88._PANEL_STORE_DONE_SCHEMA,
                    "ready": True,
                    "run_id": run_id,
                    "trade_day": day.isoformat(),
                    "manifest_path": manifest_path.resolve().as_posix(),
                    "manifest_sha256": r88._sha256(manifest_path),
                    "contract_sha256": contract_hash,
                    "assets": {asset: str(entry["sha256"]) for asset, entry in assets.items()},
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _prepared_manifest_backfill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    start_text: str | None = None,
    end_text: str | None = None,
) -> tuple[r88.PreparedR88Backfill, Path]:
    profile_path = _write_profile(tmp_path / "eligible_profile.json", _eligible_profile())
    scratch_parent = tmp_path / "research_scratch"
    scratch_root = scratch_parent / "r88_new"
    monkeypatch.setattr(r88, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    monkeypatch.setattr(r88, "_load_datahub_only_paths", _datahub_paths)
    monkeypatch.setattr(r88, "validate_rust_first_contracts", lambda _specs: None)
    monkeypatch.setattr(r88, "_assert_strict_t1429_panel_contract", lambda *_args, **_kwargs: {"lead_minutes": 1})
    rust_site = scratch_parent / "wheel_site"
    (rust_site / "cbond_on_rust").mkdir(parents=True)
    monkeypatch.setattr(r88, "_configure_isolated_rust_site", lambda _raw: rust_site)
    monkeypatch.setattr(r88, "_assert_loaded_isolated_rust_site", lambda _site: None)
    prepared = r88.preflight(
        scratch_root_text=str(scratch_root),
        profile_path=profile_path,
        start_text=start_text,
        end_text=end_text,
        rust_site_text=str(rust_site),
        verify_provenance=False,
    )
    return prepared, scratch_parent


def _write_manifest_factor_day(root: Path, day: date) -> None:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(day).replace(hour=14, minute=30), "110001.SH")],
        names=["dt", "code"],
    )
    r88.FactorStore(root, panel_name="T1430").write_day(
        day,
        pd.DataFrame({"test": [1.0]}, index=index),
    )


def _manifest_execution(*, frozen_root: Path, factor_root: Path, days: list[date]) -> dict[str, object]:
    return {
        "mode": r88._R38_EXECUTION_MODE,
        "frozen_r50_factor_root": frozen_root.resolve().as_posix(),
        "r88_factor_root": factor_root.resolve().as_posix(),
        "days": [{"day": day.isoformat()} for day in days],
    }


def test_checked_in_profile_freezes_the_complete_r88_contract_set() -> None:
    _path, profile = r88._load_profile()
    payload, pending = r88._validate_profile_structure(profile)

    assert len(payload) == 88
    assert pending == []
    assert [item["name"] for item in payload] == profile["factors"]
    assert profile["execution_status"] == r88._ELIGIBLE_STATUS
    assert profile["model_training_ready"] is False
    assert tuple(profile["research_factor_modules"]) == r88.R88_RESEARCH_FACTOR_MODULES
    assert profile["research_kernel_modules"] == dict(r88.R88_RESEARCH_FACTOR_MODULE_MAP)
    assert set(r88.R88_RESEARCH_FACTOR_MODULE_MAP).issubset(
        {item["factor"] for item in payload}
    )
    admitted = {
        str(item["name"]): str(item["rust_contract_id"])
        for item in payload
        if str(item.get("rust_contract_id") or "").startswith(f"{_R88_FIRST_BATCH_NAMESPACE}/")
    }
    assert len(admitted) == 38
    assert all(contract_id == f"{_R88_FIRST_BATCH_NAMESPACE}/{name}/v1" for name, contract_id in admitted.items())
    assert all(item.get("rust_contract_id") for item in payload)


def test_missing_rust_contracts_in_a_tampered_profile_fail_before_source_profile_or_scratch_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    scratch_root = scratch_parent / "future_r88"
    monkeypatch.setattr(r88, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    _path, profile = r88._load_profile()
    profile = dict(profile)
    specs = [dict(spec) for spec in profile["factor_specs"]]
    specs[0] = dict(specs[0])
    specs[0]["rust_contract_id"] = None
    profile["factor_specs"] = specs
    profile["pending_rust_contract_factors"] = [str(specs[0]["name"])]
    profile["execution_status"] = r88._BLOCKED_STATUS
    profile["next_execution_status"] = r88._ELIGIBLE_STATUS
    profile["specs_sha256"] = _canonical_digest(specs)
    profile_path = _write_profile(tmp_path / "missing_contract_profile.json", profile)
    monkeypatch.setattr(
        r88,
        "_load_datahub_only_paths",
        lambda: pytest.fail("blocked R88 profile must fail before DataHub path resolution"),
    )

    with pytest.raises(r88.R88RustContractIncomplete, match="1 exact rust_contract_id"):
        r88.preflight(scratch_root_text=str(scratch_root), profile_path=profile_path)
    assert not scratch_root.exists()


def test_scratch_root_must_be_a_fresh_strict_research_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(r88, "_RESEARCH_SCRATCH_PARENT", scratch_parent)

    with pytest.raises(ValueError, match="strictly below"):
        r88._assert_scratch_root(scratch_parent, execute=False)

    root = scratch_parent / "candidate"
    assert r88._assert_scratch_root(root, execute=False) == root.resolve()
    root.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="previously absent"):
        r88._assert_scratch_root(root, execute=True)


def test_profile_digest_and_order_reject_tampering() -> None:
    _path, profile = r88._load_profile()
    tampered = dict(profile)
    tampered["factors"] = list(reversed(profile["factors"]))

    with pytest.raises(r88.R88ContractError, match="ordered factor_specs names"):
        r88._validate_profile_structure(tampered)


def test_eligible_profile_redirects_every_derived_root_to_fresh_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile_path = _write_profile(tmp_path / "eligible_profile.json", _eligible_profile())
    scratch_parent = tmp_path / "research_scratch"
    scratch_root = scratch_parent / "r88_new"
    monkeypatch.setattr(r88, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    monkeypatch.setattr(r88, "_load_datahub_only_paths", _datahub_paths)
    monkeypatch.setattr(r88, "validate_rust_first_contracts", lambda specs: None)
    monkeypatch.setattr(r88, "_assert_strict_t1429_panel_contract", lambda *_args, **_kwargs: {"lead_minutes": 1})
    rust_site = scratch_parent / "wheel_site"
    (rust_site / "cbond_on_rust").mkdir(parents=True)
    monkeypatch.setattr(r88, "_configure_isolated_rust_site", lambda _raw: rust_site)
    monkeypatch.setattr(r88, "_assert_loaded_isolated_rust_site", lambda _site: None)

    prepared = r88.preflight(
        scratch_root_text=str(scratch_root),
        profile_path=profile_path,
        rust_site_text=str(rust_site),
        verify_provenance=False,
    )

    assert prepared.paths_cfg["raw_data_root"] == r88._DATAHUB_RAW_ROOT.resolve().as_posix()
    assert prepared.paths_cfg["clean_data_root"] == r88._DATAHUB_CLEAN_ROOT.resolve().as_posix()
    for field, relative in r88._DERIVED_ROOTS.items():
        assert Path(prepared.paths_cfg[field]).resolve() == (scratch_root / relative).resolve()
    assert prepared.factor_cfg["factor_files"] == []
    assert len(prepared.factor_cfg["factors"]) == 88
    assert prepared.factor_cfg["research_factor_admission_profile"] == r88._PROFILE_ID
    assert tuple(prepared.factor_cfg["research_factor_modules"]) == r88.R88_RESEARCH_FACTOR_MODULES
    assert prepared.factor_cfg["research_factor_module_map"] == dict(r88.R88_RESEARCH_FACTOR_MODULE_MAP)
    assert not scratch_root.exists()


def test_full_current_preflight_binds_clean_direct_datahub_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile_path = _write_profile(tmp_path / "eligible_profile.json", _eligible_profile())
    scratch_parent = tmp_path / "research_scratch"
    scratch_root = scratch_parent / "r88_full_current"
    monkeypatch.setattr(r88, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    monkeypatch.setattr(r88, "_load_datahub_only_paths", _datahub_paths)
    monkeypatch.setattr(r88, "validate_rust_first_contracts", lambda _specs: None)
    rust_site = scratch_parent / "wheel_site"
    (rust_site / "cbond_on_rust").mkdir(parents=True)
    monkeypatch.setattr(r88, "_configure_isolated_rust_site", lambda _raw: rust_site)
    monkeypatch.setattr(r88, "_assert_loaded_isolated_rust_site", lambda _site: None)

    prepared = r88.preflight(
        scratch_root_text=str(scratch_root),
        profile_path=profile_path,
        rust_site_text=str(rust_site),
        verify_provenance=False,
        full_current_r88=True,
    )

    assert Path(prepared.paths_cfg["panel_data_root"]).resolve() == r88._DATAHUB_CLEAN_ROOT.resolve()
    assert prepared.factor_cfg["panel_source"] == {"mode": "clean_direct"}
    panel_cfg = prepared.factor_cfg["panel_build_config"]
    assert panel_cfg["panel_name"] == "T1430"
    assert panel_cfg["lead_minutes"] == 1
    assert panel_cfg["schedule"]["windows"] == [{"start": "14:30", "end": "14:30"}]
    assert set(prepared.paths_cfg) == {
        "raw_data_root",
        "clean_data_root",
        "cleaned_data_root",
        "panel_data_root",
        "factor_data_root",
        "results_root",
        "factor_table",
    }
    assert prepared.experiment_generation_id == r88._FULL_CURRENT_DEFAULT_GENERATION_ID
    assert prepared.start == r88._FULL_CURRENT_EXECUTABLE_START
    assert prepared.end == r88._FULL_CURRENT_EXECUTABLE_END
    assert Path(prepared.paths_cfg["factor_data_root"]).resolve() == (
        r88._CANONICAL_EXPERIMENT_TABLE_ROOT / "generations" / r88._FULL_CURRENT_DEFAULT_GENERATION_ID
    ).resolve()
    assert prepared.paths_cfg["factor_table"] == {
        "table_id": "experiment",
        "root": r88._CANONICAL_FACTOR_STORE_ROOT.resolve().as_posix(),
        "generation_id": r88._FULL_CURRENT_DEFAULT_GENERATION_ID,
    }
    assert Path(prepared.paths_cfg["results_root"]).resolve() == (scratch_root / "results").resolve()
    assert not scratch_root.exists()


def test_normal_cli_rejects_legacy_r38_writer_before_any_preflight(tmp_path: Path) -> None:
    scratch_root = tmp_path / "research_scratch" / "legacy_r38"
    with pytest.raises(SystemExit):
        r88.main(["--scratch-root", str(scratch_root), "--execute"])
    assert not scratch_root.exists()


def test_full_current_execute_requires_an_explicit_generation_before_preflight(tmp_path: Path) -> None:
    scratch_root = tmp_path / "research_scratch" / "missing_generation"

    with pytest.raises(SystemExit):
        r88.main(
            [
                "--scratch-root",
                str(scratch_root),
                "--full-current-r88",
                "--execute",
            ]
        )

    assert not scratch_root.exists()


def test_completed_manifest_binds_the_profile_factor_root_and_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared, scratch_parent = _prepared_manifest_backfill(monkeypatch, tmp_path)
    factor_root = Path(prepared.paths_cfg["factor_data_root"])
    result_root = Path(prepared.paths_cfg["results_root"]) / "mocked"
    result_root.mkdir(parents=True)
    frozen_root = scratch_parent / "frozen_r50"
    days = [date(2024, 1, 3), date(2026, 7, 30)]
    for day in days:
        _write_manifest_factor_day(frozen_root, day)
        _write_manifest_factor_day(factor_root, day)
    monkeypatch.setattr(r88, "_FROZEN_R50_FACTOR_ROOT", frozen_root)

    manifest_path = r88._write_backfill_manifest(
        prepared=prepared,
        result_root=result_root,
        execution=_manifest_execution(frozen_root=frozen_root, factor_root=factor_root, days=days),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == r88._BACKFILL_MANIFEST_SCHEMA
    assert payload["execution_status"] == r88._FULL_BACKFILL_STATUS
    assert payload["model_training_ready"] is True
    assert payload["factor_root"] == factor_root.resolve().as_posix()
    assert payload["factors"] == prepared.profile["factors"]
    assert payload["profile"]["specs_sha256"] == prepared.profile["specs_sha256"]
    assert payload["research_module_admission"]["modules"] == list(
        r88.R88_RESEARCH_FACTOR_MODULES
    )
    assert payload["research_module_admission"]["factor_key_to_module"] == dict(
        r88.R88_RESEARCH_FACTOR_MODULE_MAP
    )
    assert payload["completion"]["full_completion"] is True
    assert payload["completion"]["expected_frozen_r50_days"] == [day.isoformat() for day in days]
    assert payload["completion"]["execution_days"] == [day.isoformat() for day in days]
    assert payload["completion"]["r88_factor_store_days"] == [day.isoformat() for day in days]


def test_partial_smoke_manifest_cannot_admit_model_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_day = date(2026, 7, 30)
    prepared, scratch_parent = _prepared_manifest_backfill(
        monkeypatch,
        tmp_path,
        start_text=smoke_day.isoformat(),
        end_text=smoke_day.isoformat(),
    )
    factor_root = Path(prepared.paths_cfg["factor_data_root"])
    result_root = Path(prepared.paths_cfg["results_root"]) / "smoke"
    result_root.mkdir(parents=True)
    frozen_root = scratch_parent / "frozen_r50"
    _write_manifest_factor_day(frozen_root, smoke_day)
    _write_manifest_factor_day(factor_root, smoke_day)
    monkeypatch.setattr(r88, "_FROZEN_R50_FACTOR_ROOT", frozen_root)

    manifest_path = r88._write_backfill_manifest(
        prepared=prepared,
        result_root=result_root,
        execution=_manifest_execution(
            frozen_root=frozen_root,
            factor_root=factor_root,
            days=[smoke_day],
        ),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["execution_status"] == r88._PARTIAL_BACKFILL_STATUS
    assert payload["model_training_ready"] is False
    assert payload["completion"]["full_completion"] is False
    assert "requested range is not the full approved R88 range" in payload["completion"]["blocking_reasons"]


def test_full_range_manifest_requires_exact_frozen_r50_day_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared, scratch_parent = _prepared_manifest_backfill(monkeypatch, tmp_path)
    factor_root = Path(prepared.paths_cfg["factor_data_root"])
    result_root = Path(prepared.paths_cfg["results_root"]) / "incomplete"
    result_root.mkdir(parents=True)
    frozen_root = scratch_parent / "frozen_r50"
    days = [date(2024, 1, 3), date(2026, 7, 30)]
    for day in days:
        _write_manifest_factor_day(frozen_root, day)
    _write_manifest_factor_day(factor_root, days[0])
    monkeypatch.setattr(r88, "_FROZEN_R50_FACTOR_ROOT", frozen_root)

    manifest_path = r88._write_backfill_manifest(
        prepared=prepared,
        result_root=result_root,
        execution=_manifest_execution(frozen_root=frozen_root, factor_root=factor_root, days=days),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["execution_status"] == r88._PARTIAL_BACKFILL_STATUS
    assert payload["model_training_ready"] is False
    assert payload["completion"]["full_completion"] is False
    assert (
        "durable R88 FactorStore days do not exactly equal frozen R50 approved-range days"
        in payload["completion"]["blocking_reasons"]
    )


def test_eligible_profile_requires_a_scratch_wheel_before_datahub_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile_path = _write_profile(tmp_path / "eligible_profile.json", _eligible_profile())
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(r88, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    monkeypatch.setattr(
        r88,
        "_load_datahub_only_paths",
        lambda: pytest.fail("eligible R88 without an isolated wheel must fail before DataHub path resolution"),
    )
    monkeypatch.setattr(r88, "validate_rust_first_contracts", lambda specs: None)

    with pytest.raises(r88.R88ContractError, match="requires --rust-site"):
        r88.preflight(
            scratch_root_text=str(scratch_parent / "candidate"),
            profile_path=profile_path,
            verify_provenance=False,
        )


def test_r88_requires_the_strict_1429_panel_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        r88,
        "load_config_file",
        lambda _name: {
            "panel_name": "T1430",
            "lead_minutes": 0,
            "schedule": {"mode": "custom_windows", "windows": [{"start": "14:30", "end": "14:30"}]},
        },
    )
    with pytest.raises(r88.R88ContractError, match="lead_minutes=1"):
        r88._assert_strict_t1429_panel_contract()


def _factor_index(day: date, codes: list[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [[pd.Timestamp(day).replace(hour=14, minute=30)] * len(codes), codes],
        names=["dt", "code"],
    )


def test_profile_splits_exactly_into_frozen_r50_and_new_r38() -> None:
    _path, profile = r88._load_profile()
    payload, _pending = r88._validate_profile_structure(profile)
    r50, r38 = r88._split_frozen_r50_and_r38_specs(r88._to_specs(payload))

    assert len(r50) == 50
    assert len(r38) == 38
    assert all(str(spec.rust_contract_id).startswith("live50_r5/") for spec in r50)
    assert all(str(spec.rust_contract_id).startswith("research_r88_20260825/") for spec in r38)
    assert not set(r88.build_factor_col(spec) for spec in r50).intersection(
        r88.build_factor_col(spec) for spec in r38
    )


def test_exact_merge_rejects_index_drift_and_preserves_profile_order() -> None:
    day = date(2026, 1, 2)
    r50_cols = [f"old_{index}" for index in range(50)]
    r38_cols = [f"new_{index}" for index in range(38)]
    index = _factor_index(day, ["110001.SH", "110002.SH"])
    r50 = pd.DataFrame(1.0, index=index, columns=r50_cols)
    r38 = pd.DataFrame(2.0, index=index, columns=r38_cols)
    order = [r50_cols[1], r38_cols[1], r50_cols[0], *r50_cols[2:], r38_cols[0], *r38_cols[2:]]

    merged = r88._merge_exact_r50_r38(
        r50,
        r38,
        day=day,
        r50_columns=r50_cols,
        r38_columns=r38_cols,
        output_columns=order,
    )
    assert merged.index.equals(index)
    assert list(merged.columns) == order
    assert merged.loc[:, r50_cols].equals(r50)

    with pytest.raises(r88.R88ContractError, match="exactly equal"):
        r88._merge_exact_r50_r38(
            r50,
            r38.iloc[:1],
            day=day,
            r50_columns=r50_cols,
            r38_columns=r38_cols,
            output_columns=order,
        )


def test_strict_current_day_panel_uses_r50_codes_and_filters_lunch_and_post_cutoff(
    tmp_path: Path,
) -> None:
    day = date(2026, 1, 2)
    snapshot = tmp_path / "snapshot.parquet"
    pd.DataFrame(
        {
            "code": ["110001.SH", "110001.SH", "110001.SH", "110002.SH", "110002.SH", "999999.SH"],
            "trade_time": [
                "2026-01-02 09:30:00",
                "2026-01-02 11:30:00",
                "2026-01-02 14:29:00.000000999",
                "2026-01-02 13:00:00",
                "2026-01-02 14:30:00",
                "2026-01-02 14:00:00",
            ],
            "last": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    ).to_parquet(snapshot, index=False)
    index = _factor_index(day, ["110001.SH", "110002.SH"])
    panel = r88._strict_current_day_panel(
        snapshot,
        day=day,
        eligible_index=index,
        columns=["code", "trade_time", "last"],
    )

    assert set(panel.index.get_level_values("code")) == {"110001.SH", "110002.SH"}
    assert len(panel) == 3
    assert panel.attrs["__build_day__"] == "2026-01-02"
    assert panel.index.get_level_values("seq").tolist() == [0, 1, 0]


def test_r38_only_execute_writes_fresh_intermediate_and_final_factorstores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    frozen_root = scratch_parent / "frozen_r50"
    day = date(2026, 1, 2)
    r50_cols = [f"old_{index}" for index in range(50)]
    r38_cols = [f"new_{index}" for index in range(38)]
    specs = tuple(
        [
            r88.FactorSpec(name=column, factor="f", rust_contract_id=f"live50_r5/{column}")
            for column in r50_cols
        ]
        + [
            r88.FactorSpec(
                name=column,
                factor="f",
                rust_contract_id=f"research_r88_20260825/{column}/v1",
            )
            for column in r38_cols
        ]
    )
    r50, r38 = r88._split_frozen_r50_and_r38_specs(specs)
    index = _factor_index(day, ["110001.SH", "110002.SH"])
    frozen_store = r88.FactorStore(frozen_root, panel_name="T1430")
    frozen_store.write_day(day, pd.DataFrame(1.0, index=index, columns=r50_cols))
    run_root = scratch_parent / "run"
    prepared = r88.PreparedR88Backfill(
        profile_path=tmp_path / "profile.json",
        profile={"factor_specs": []},
        specs=specs,
        frozen_r50_specs=r50,
        r38_specs=r38,
        scratch_root=run_root,
        paths_cfg={
            "raw_data_root": str(tmp_path / "raw"),
            "cleaned_data_root": str(tmp_path / "clean"),
            "factor_data_root": str(run_root / "factor_data"),
            "results_root": str(run_root / "results"),
        },
        factor_cfg={},
        start=day,
        end=day,
    )
    (tmp_path / "raw").mkdir()
    (tmp_path / "clean").mkdir()
    monkeypatch.setattr(r88, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    monkeypatch.setattr(r88, "_FROZEN_R50_FACTOR_ROOT", frozen_root)
    monkeypatch.setattr(r88, "prepare_factor_modules", lambda _cfg: None)
    monkeypatch.setattr(r88, "validate_rust_first_contracts", lambda _specs: None)
    monkeypatch.setattr(
        r88,
        "_prepare_r38_context_loaders",
        lambda **_kwargs: r88.R38ContextLoaders((), {}, {}, False, None),
    )
    monkeypatch.setattr(r88, "_load_r38_daily_and_map_context", lambda **_kwargs: ({}, None))
    monkeypatch.setattr(r88, "_mapped_stock_codes_for_r38", lambda **_kwargs: {"600000.SH"})
    fake_panel = pd.DataFrame({"last": [100.0, 101.0]}, index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp(day).replace(hour=14, minute=30), "110001.SH", 0), (pd.Timestamp(day).replace(hour=14, minute=30), "110002.SH", 0)],
        names=["dt", "code", "seq"],
    ))
    fake_panel.attrs["__build_day__"] = day.isoformat()
    monkeypatch.setattr(r88, "_strict_current_day_panel", lambda *_args, **_kwargs: fake_panel)
    monkeypatch.setattr(
        r88,
        "build_factor_frame_rust",
        lambda _panel, specs, **_kwargs: pd.DataFrame(2.0, index=index, columns=[r88.build_factor_col(spec) for spec in specs]),
    )

    result_root, execution = r88._execute_r38_only_backfill(prepared)
    assert execution["mode"] == r88._R38_EXECUTION_MODE
    assert (result_root / "r38_execution_manifest.json").is_file()
    assert r88.FactorStore(run_root / r88._R38_FACTOR_STORE_DIR, panel_name="T1430").read_day(day).shape == (2, 38)
    assert r88.FactorStore(run_root / "factor_data", panel_name="T1430").read_day(day).shape == (2, 88)


def test_r38_only_execute_keeps_r50_row_and_marks_missing_current_snapshot_r38_nan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    frozen_root = scratch_parent / "frozen_r50"
    day = date(2026, 1, 2)
    r50_cols = [f"old_{index}" for index in range(50)]
    r38_cols = [f"new_{index}" for index in range(38)]
    specs = tuple(
        [
            r88.FactorSpec(name=column, factor="f", rust_contract_id=f"live50_r5/{column}")
            for column in r50_cols
        ]
        + [
            r88.FactorSpec(
                name=column,
                factor="f",
                rust_contract_id=f"research_r88_20260825/{column}/v1",
            )
            for column in r38_cols
        ]
    )
    r50, r38 = r88._split_frozen_r50_and_r38_specs(specs)
    full_index = _factor_index(day, ["110001.SH", "110002.SH"])
    observed_index = _factor_index(day, ["110001.SH"])
    r88.FactorStore(frozen_root, panel_name="T1430").write_day(
        day,
        pd.DataFrame(1.0, index=full_index, columns=r50_cols),
    )
    run_root = scratch_parent / "run_missing"
    prepared = r88.PreparedR88Backfill(
        profile_path=tmp_path / "profile.json",
        profile={"factor_specs": []},
        specs=specs,
        frozen_r50_specs=r50,
        r38_specs=r38,
        scratch_root=run_root,
        paths_cfg={
            "raw_data_root": str(tmp_path / "raw"),
            "cleaned_data_root": str(tmp_path / "clean"),
            "factor_data_root": str(run_root / "factor_data"),
            "results_root": str(run_root / "results"),
        },
        factor_cfg={},
        start=day,
        end=day,
    )
    (tmp_path / "raw").mkdir()
    (tmp_path / "clean").mkdir()
    monkeypatch.setattr(r88, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    monkeypatch.setattr(r88, "_FROZEN_R50_FACTOR_ROOT", frozen_root)
    monkeypatch.setattr(r88, "prepare_factor_modules", lambda _cfg: None)
    monkeypatch.setattr(r88, "validate_rust_first_contracts", lambda _specs: None)
    monkeypatch.setattr(
        r88,
        "_prepare_r38_context_loaders",
        lambda **_kwargs: r88.R38ContextLoaders((), {}, {}, False, None),
    )
    monkeypatch.setattr(r88, "_load_r38_daily_and_map_context", lambda **_kwargs: ({}, None))
    monkeypatch.setattr(r88, "_mapped_stock_codes_for_r38", lambda **_kwargs: {"600000.SH"})
    bond_panel = pd.DataFrame(
        {"last": [100.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp(day).replace(hour=14, minute=30), "110001.SH", 0)],
            names=["dt", "code", "seq"],
        ),
    )
    bond_panel.attrs["__build_day__"] = day.isoformat()
    stock_panel = bond_panel.rename(index={"110001.SH": "600000.SH"}, level="code")
    calls = iter([bond_panel, stock_panel])
    monkeypatch.setattr(r88, "_strict_current_day_panel", lambda *_args, **_kwargs: next(calls))
    monkeypatch.setattr(
        r88,
        "build_factor_frame_rust",
        lambda _panel, specs, **_kwargs: pd.DataFrame(
            2.0,
            index=observed_index,
            columns=[r88.build_factor_col(spec) for spec in specs],
        ),
    )

    _result_root, execution = r88._execute_r38_only_backfill(prepared)
    day_row = execution["days"][0]
    assert day_row["source_missing_bond_code_count"] == 1
    assert day_row["source_missing_bond_codes"] == ["110002.SH"]
    final = r88.FactorStore(run_root / "factor_data", panel_name="T1430").read_day(day)
    assert final.index.equals(full_index)
    assert final.loc[(pd.Timestamp(day).replace(hour=14, minute=30), "110002.SH"), r50_cols].eq(1.0).all()
    assert final.loc[(pd.Timestamp(day).replace(hour=14, minute=30), "110002.SH"), r38_cols].isna().all()


def test_full_current_datahub_calendar_requires_raw_clean_cbond_stock_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    days = [date(2026, 1, 2), date(2026, 1, 5)]
    monkeypatch.setattr(r88, "list_trading_days_from_raw", lambda *_args, **_kwargs: list(days))
    monkeypatch.setattr(
        r88,
        "_iter_existing_snapshot_days",
        lambda *_args, **_kwargs: list(days),
    )

    actual, contract = r88._full_current_datahub_calendar(
        raw_data_root=tmp_path / "raw",
        cleaned_data_root=tmp_path / "clean",
        start=days[0],
        end=days[-1],
    )

    assert actual == days
    assert contract["source"] == r88._FULL_CURRENT_CALENDAR_SOURCE
    assert contract["days_sha256"] == r88._days_sha256(days)
    assert contract["clean_cbond_snapshot_days"] == [day.isoformat() for day in days]

    def _mismatched_clean(*_args: object, **kwargs: object) -> list[date]:
        return [days[0]] if kwargs.get("asset") == "stock" else list(days)

    monkeypatch.setattr(r88, "_iter_existing_snapshot_days", _mismatched_clean)
    with pytest.raises(r88.R88ContractError, match="clean stock snapshot calendars differ"):
        r88._full_current_datahub_calendar(
            raw_data_root=tmp_path / "raw",
            cleaned_data_root=tmp_path / "clean",
            start=days[0],
            end=days[-1],
        )


def test_full_current_panel_consumer_requires_committed_cbond_and_stock_bundles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    days = [date(2026, 1, 2), date(2026, 1, 5)]
    public_root = tmp_path / "public_panel_store"
    datahub_manifest_root = tmp_path / "datahub_manifests"
    _write_published_panel_store(public_root, days=days, datahub_manifest_root=datahub_manifest_root)
    monkeypatch.setattr(r88, "_PUBLIC_PANEL_STORE_ROOT", public_root)
    monkeypatch.setattr(r88, "_DATAHUB_MANIFEST_ROOT", datahub_manifest_root)

    evidence = r88._verify_published_panel_store(public_root, expected_days=days)

    assert evidence["root"] == public_root.resolve().as_posix()
    assert evidence["days_sha256"] == r88._days_sha256(days)
    assert [row["day"] for row in evidence["days"]] == [day.isoformat() for day in days]
    assert set(evidence["days"][0]["assets"]) == {"cbond", "stock"}

    missing_stock = r88._panel_store_asset_path(public_root, days[0], "stock")
    missing_stock.unlink()
    with pytest.raises(r88.R88ContractError, match="stock parquet calendar"):
        r88._verify_published_panel_store(public_root, expected_days=days)


def test_full_current_panel_consumer_rejects_done_manifest_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    day = date(2026, 1, 2)
    public_root = tmp_path / "public_panel_store"
    datahub_manifest_root = tmp_path / "datahub_manifests"
    _write_published_panel_store(public_root, days=[day], datahub_manifest_root=datahub_manifest_root)
    monkeypatch.setattr(r88, "_PUBLIC_PANEL_STORE_ROOT", public_root)
    monkeypatch.setattr(r88, "_DATAHUB_MANIFEST_ROOT", datahub_manifest_root)
    done_path = r88._panel_store_done_path(public_root, day)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["manifest_sha256"] = "0" * 64
    done_path.write_text(json.dumps(done) + "\n", encoding="utf-8")

    with pytest.raises(r88.R88ContractError, match="done manifest hash mismatch"):
        r88._verify_published_panel_store(public_root, expected_days=[day])


def test_full_current_execution_uses_in_memory_clean_direct_panels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    day = date(2026, 1, 2)
    scratch_root = tmp_path / "research_scratch" / "r88_current"
    raw_root = tmp_path / "raw"
    clean_root = tmp_path / "clean"
    raw_root.mkdir()
    clean_root.mkdir()
    canonical_root = tmp_path / "factor_store"
    CanonicalFactorStore(canonical_root).initialize()
    canonical_experiment_root = canonical_root / "experiment"
    generation_id = "r88-current-test"
    generation_root = canonical_experiment_root / "generations" / generation_id
    canonical_contract = FactorTableContract(
        (FactorColumnContract("alpha", "v1", "a" * 64, "alpha"),)
    )
    legacy_timestamp = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    CanonicalFactorStore(canonical_root).write_day(
        "experiment",
        day,
        pd.DataFrame(
            {"alpha": [9.0]},
            index=pd.MultiIndex.from_tuples([(legacy_timestamp, "110001.SH")], names=["dt", "code"]),
        ),
        contract=canonical_contract,
    )
    profile = _eligible_profile()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    prepared = r88.PreparedR88Backfill(
        profile_path=profile_path,
        profile=profile,
        specs=(),
        frozen_r50_specs=(),
        r38_specs=(),
        scratch_root=scratch_root,
        paths_cfg={
            "raw_data_root": str(raw_root),
            "cleaned_data_root": str(clean_root),
            "panel_data_root": str(clean_root),
            "factor_data_root": str(generation_root),
            "factor_table": {"table_id": "experiment", "root": str(canonical_root), "generation_id": generation_id},
            "results_root": str(scratch_root / "results"),
        },
        factor_cfg={
            "panel_source": {"mode": "clean_direct"},
            "panel_build_config": {
                "panel_name": "T1430",
                "panel_mode": "snapshot_sequence",
                "assets": ["cbond", "stock"],
                "lead_minutes": 1,
                "count_points": 5000,
                "max_lookback_days": 4,
                "schedule": {"mode": "custom_windows", "windows": [{"start": "14:30", "end": "14:30"}]},
                "snapshot": {},
            },
            "context": {},
            "compute": {},
        },
        start=day,
        end=day,
        experiment_generation_id=generation_id,
    )
    calendar = {
        "source": r88._FULL_CURRENT_CALENDAR_SOURCE,
        "raw_cbond_snapshot_days": [day.isoformat()],
        "raw_stock_snapshot_days": [day.isoformat()],
        "clean_cbond_snapshot_days": [day.isoformat()],
        "clean_stock_snapshot_days": [day.isoformat()],
        "days_sha256": r88._days_sha256([day]),
    }
    direct_evidence = {
        "mode": "clean_direct",
        "manifest_root": str(tmp_path / "manifests"),
        "days_sha256": r88._days_sha256([day]),
        "days": [
            {
                "day": day.isoformat(),
                "run_id": "run",
                "clean_manifest": "clean.json",
                "clean_manifest_sha256": "a" * 64,
                "publish_done": "done.json",
                "publish_done_sha256": "b" * 64,
            }
        ],
    }
    monkeypatch.setattr(r88, "_CANONICAL_FACTOR_STORE_ROOT", canonical_root)
    monkeypatch.setattr(r88, "_CANONICAL_EXPERIMENT_TABLE_ROOT", canonical_experiment_root)
    monkeypatch.setattr(r88, "_full_current_datahub_calendar", lambda **_kwargs: ([day], calendar))
    monkeypatch.setattr(r88, "_verify_direct_datahub_evidence", lambda **_kwargs: direct_evidence)
    monkeypatch.setattr(
        r88,
        "_verify_published_panel_store",
        lambda *_args, **_kwargs: pytest.fail("clean-direct full-current execution must not read PanelStore"),
    )
    monkeypatch.setattr(r88, "prepare_factor_modules", lambda _cfg: None)
    monkeypatch.setattr(r88, "validate_rust_first_contracts", lambda _specs: None)
    monkeypatch.setattr(r88, "current_catalog_contract", lambda _specs: canonical_contract)
    monkeypatch.setattr(
        r88,
        "issue_experiment_publisher_authority",
        lambda *, root, research_only, generation_id: issue_test_canonical_writer_authority(
            root=root,
            table_id="experiment",
            generation_id=generation_id,
        ),
    )
    captured: dict[str, object] = {}

    def _run_factor_pipeline(panel_data_root: object, factor_data_root: object, *_args: object, **kwargs: object) -> SimpleNamespace:
        captured["panel_data_root"] = panel_data_root
        captured["factor_data_root"] = factor_data_root
        captured.update(kwargs)
        writer = kwargs["factor_store"]
        assert isinstance(writer, r88.CanonicalFactorTableWriter)
        writer.write_day(
            day,
            pd.DataFrame(
                {"alpha": [1.0]},
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30), "110001.SH")],
                    names=["dt", "code"],
                ),
            ),
        )
        return SimpleNamespace(written=1, skipped=0)

    monkeypatch.setattr(r88, "run_factor_pipeline", _run_factor_pipeline)
    monkeypatch.setattr(
        r88,
        "_full_current_execution_ledger",
        lambda **_kwargs: (
            [
                {
                    "day": day.isoformat(),
                    "inf_cell_count": 0,
                    "eligible_66of88_row_count": 1,
                    "all_nan_columns": [],
                }
            ],
            [day],
        ),
    )

    result_root, execution = r88._execute_full_current_r88_recompute(
        prepared,
        workers=1,
        factor_workers=1,
    )

    assert Path(captured["panel_data_root"]).resolve() == clean_root.resolve()
    assert captured["panel_source_cfg"] == {"mode": "clean_direct"}
    assert captured["panel_build_cfg"]["lead_minutes"] == 1
    assert captured["panel_build_cfg"]["schedule"]["windows"] == [{"start": "14:30", "end": "14:30"}]
    assert Path(captured["cleaned_data_root"]).resolve() == clean_root.resolve()
    assert Path(captured["factor_data_root"]).resolve() == generation_root.resolve()
    assert isinstance(captured["factor_store"], r88.CanonicalFactorTableWriter)
    assert not (scratch_root / "panel_data").exists()
    assert not (scratch_root / "factor_data").exists()
    assert result_root.is_dir()
    assert execution["clean_direct_panel"]["datahub_evidence"] == direct_evidence
    assert execution["experiment_generation"]["status"] == "finalized_inactive"
    assert CanonicalFactorStore(canonical_root).active_experiment_generation_id() == "legacy_flat"
    assert CanonicalFactorStore(canonical_root).read_day("experiment", day).iloc[0, 0] == 9.0
    assert CanonicalFactorStore(canonical_root).read_day(
        "experiment",
        day,
        generation_id=generation_id,
    ).iloc[0, 0] == 1.0


def test_full_current_completion_does_not_consult_frozen_r50(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    days = [date(2026, 1, 2), date(2026, 1, 5)]
    monkeypatch.setattr(r88, "_FULL_CURRENT_EXECUTABLE_START", days[0])
    monkeypatch.setattr(r88, "_FULL_CURRENT_EXECUTABLE_END", days[-1])
    monkeypatch.setattr(r88, "_FULL_CURRENT_EXPECTED_DAY_COUNT", len(days))
    monkeypatch.setattr(
        r88,
        "_assert_frozen_r50_root",
        lambda: pytest.fail("full-current completion must not read frozen R50"),
    )

    canonical_root = tmp_path / "factor_store"
    canonical = CanonicalFactorStore(canonical_root)
    canonical.initialize()
    generation_id = "r88-completion-test"
    generation = canonical.create_stage_generation(
        generation_id,
        source_evidence={"source": "r88_full_current_rust88_recompute", "fixture": True},
    )
    factor_root = generation.generation_root
    contract = FactorTableContract(
        (FactorColumnContract("alpha", "v1", "a" * 64, "alpha"),)
    )
    for day in days:
        timestamp = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
        frame = pd.DataFrame(
            {"alpha": [1.0]},
            index=pd.MultiIndex.from_tuples([(timestamp, "110001.SH")], names=["dt", "code"]),
        )
        canonical.write_day("experiment", day, frame, contract=contract, generation_id=generation_id)
    coverage = {
        "calendar_source": r88._FULL_CURRENT_CALENDAR_SOURCE,
        "score_day_start": days[0].isoformat(),
        "score_day_end": days[-1].isoformat(),
        "expected_day_count": len(days),
    }
    canonical.record_experiment_generation_attestation(
        generation_id,
        {"migration_id": "r88-test-full", "scope": "full", "expected_coverage": coverage},
    )
    canonical.mark_experiment_generation_verified(
        generation_id,
        migration_id="r88-test-full",
        coverage=coverage,
    )
    canonical.finalize_stage_generation(
        generation_id,
        expected_days=days,
        verification={"fixture": "r88-full-current"},
    )
    monkeypatch.setattr(r88, "_CANONICAL_FACTOR_STORE_ROOT", canonical_root)
    monkeypatch.setattr(r88, "_CANONICAL_EXPERIMENT_TABLE_ROOT", factor_root)
    profile = _eligible_profile()
    prepared = r88.PreparedR88Backfill(
        profile_path=tmp_path / "profile.json",
        profile=profile,
        specs=(),
        frozen_r50_specs=(),
        r38_specs=(),
        scratch_root=tmp_path / "scratch",
        paths_cfg={
            "factor_data_root": str(factor_root),
            "factor_table": {"table_id": "experiment", "root": str(canonical_root), "generation_id": generation_id},
        },
        factor_cfg={},
        start=days[0],
        end=days[-1],
        experiment_generation_id=generation_id,
    )
    calendar_days = [day.isoformat() for day in days]
    calendar = {
        "source": r88._FULL_CURRENT_CALENDAR_SOURCE,
        "raw_cbond_snapshot_days": calendar_days,
        "raw_stock_snapshot_days": calendar_days,
        "clean_cbond_snapshot_days": calendar_days,
        "clean_stock_snapshot_days": calendar_days,
        "days_sha256": r88._days_sha256(days),
    }
    execution = {
        "mode": "full_current_rust88_recompute",
        "factor_root": factor_root.resolve().as_posix(),
        "factor_table": {
            "table_id": "experiment",
            "root": canonical_root.resolve().as_posix(),
            "generation_id": generation_id,
        },
        "factor_count": 88,
        "no_frozen_factorstore_read": True,
        "calendar_source": r88._FULL_CURRENT_CALENDAR_SOURCE,
        "calendar_contract": calendar,
        "calendar_sha256": r88._days_sha256(days),
        "clean_direct_panel": {
            "mode": "clean_direct",
            "cleaned_data_root": r88._DATAHUB_CLEAN_ROOT.resolve().as_posix(),
            "panel_config_sha256": "a" * 64,
            "panel_contract": {
                "panel_name": "T1430",
                "physical_cutoff": "14:29:00.999999",
                "lead_minutes": 1,
            },
            "datahub_evidence": {
                "mode": "clean_direct",
                "days_sha256": r88._days_sha256(days),
                "days": [{"day": day.isoformat()} for day in days],
            },
        },
        "expected_calendar_days": calendar_days,
        "days": [{"day": day.isoformat()} for day in days],
        "full_calendar_completion": True,
        "experiment_generation": {
            "generation_id": generation_id,
            "status": "finalized_inactive",
            "active_generation_before": "legacy_flat",
            "active_generation_after": "legacy_flat",
        },
    }

    completion = r88._completion_evidence(prepared=prepared, execution=execution)

    assert completion["definition"] == "full_current_rust88_profile_recompute_exact_datahub_calendar"
    assert completion["full_completion"] is True
