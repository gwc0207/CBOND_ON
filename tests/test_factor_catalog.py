from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys

from cbond_on.domain.factor_catalog import (
    load_factor_catalog,
    load_live_release,
    resolve_factor_instance,
    resolve_operator_modules,
    validate_factor_catalog,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_definition(path: Path, *, factor_id: str):
    module_name = f"_test_catalog_definition_{factor_id}"
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_factor_catalog_is_complete_and_internally_consistent() -> None:
    summary = validate_factor_catalog(REPO_ROOT)

    assert summary == {
        "factor_count": 800,
        "operator_count": 266,
        "family_count": 174,
        "manual_override_count": 1,
        "live_release_count": 1,
        "live_released_factor_count": 50,
    }

    catalog = load_factor_catalog(REPO_ROOT)
    assert len(catalog) == 800
    assert sum(row["source_set"] == "research773" for row in catalog.values()) == 773
    assert sum(row["source_set"] == "legacy_live27" for row in catalog.values()) == 27


def test_same_name_drrc_history_is_explicitly_overridden() -> None:
    path = REPO_ROOT / "factor_engine" / "catalog" / "manual_overrides.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides = {item["factor_id"]: item for item in payload["overrides"]}

    override = overrides["drrc_return_amount_rank_spearman60"]
    assert override["operator_id"] == "factor_mining_daily_relative_rank_flow_coupling_v2"
    assert override["primary_family"] == "prior_relative_return_flow_rank_coupling"
    assert "source_position=3" in override["reason"]


def test_drrc_name_collision_fails_closed_without_its_manual_override(tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "schema_version": "factor_catalog_manual_overrides/v1",
                "overrides": [],
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "harness/tools/build_factor_catalog.py",
            "--overrides",
            str(overrides),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "ambiguous legacy mapping requires explicit override" in result.stderr
    assert "drrc_return_amount_rank_spearman60" in result.stderr


def test_catalog_builder_check_is_reproducible_without_external_scratch_inputs() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "harness/tools/build_factor_catalog.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "check"
    assert payload["factor_count"] == 800
    assert payload["operator_count"] == 266


def test_public_catalog_resolver_is_metadata_only_and_live50_is_exactly_bound() -> None:
    legacy = resolve_factor_instance("range_30m", repo_root=REPO_ROOT)
    assert legacy["source_set"] == "legacy_live27"
    assert legacy["operator_id"] == "range_ratio"

    research = resolve_factor_instance("base_debt_premium_floor_gap", repo_root=REPO_ROOT)
    assert research["source_set"] == "research773"
    assert research["operator_id"] == "factor_mining_daily_catalog_v1"

    operators = resolve_operator_modules(
        ["range_30m", "base_debt_premium_floor_gap"],
        repo_root=REPO_ROOT,
    )
    assert {row["operator_id"] for row in operators} == {
        "range_ratio",
        "factor_mining_daily_catalog_v1",
    }
    assert all(row["availability_status"] == "registered_static_operator" for row in operators)

    release = load_live_release("live50_rust50_operator_source_20260826", repo_root=REPO_ROOT)
    assert release["factor_count"] == 50
    assert [row["position"] for row in release["instances"]] == list(range(1, 51))
    assert release["instances"][0]["factor_id"] == "cb_overnight_return_mean_20d"
    assert release["instances"][0]["rust_contract_id"] == "live50_r5/cb_overnight_return_mean_20d"
    assert all(
        resolve_factor_instance(row["factor_id"], repo_root=REPO_ROOT)["lifecycle"]
        == {
            "catalog_status": "registered",
            "live_admission_status": "live_released",
            "rust_status": "live50_rust_capability_required",
        }
        for row in release["instances"]
    )


def test_all_800_parameterized_definition_entries_build_exact_factor_specs() -> None:
    catalog = load_factor_catalog(REPO_ROOT)

    for factor_id, row in catalog.items():
        module = _load_definition(
            REPO_ROOT / "factor_engine" / row["definition_path"],
            factor_id=factor_id,
        )
        payload = module.definition_payload()
        spec = module.build_factor_spec()
        contract = json.loads(
            (REPO_ROOT / "factor_engine" / row["contract_path"]).read_text(encoding="utf-8")
        )

        assert payload == {
            "factor_id": factor_id,
            "factor_version": row["factor_version"],
            "primary_family": row["primary_family"],
            "operator_id": row["operator_id"],
            "fixed_params": contract["fixed_params"],
            "output_col": None,
            "rust_contract_id": row["rust_contract_id"],
            "operator_contract_path": contract["operator_bindings"][0]["operator_contract_path"],
        }
        assert spec.name == factor_id
        assert spec.factor == row["operator_id"]
        assert spec.params == contract["fixed_params"]
        assert spec.output_col is None
        assert spec.rust_contract_id == row["rust_contract_id"]


def test_public_catalog_import_never_imports_runtime_operators() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; import cbond_on.domain.factor_catalog as c; "
                "c.load_factor_catalog(); "
                "assert not any(name.startswith('cbond_on.domain.factors.operators') "
                "for name in sys.modules)"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_definition_entry_can_build_a_spec_without_importing_runtime_operators() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import importlib.util, sys; "
                "p='factor_engine/factors/intraday_range/range_30m/definition.py'; "
                "s=importlib.util.spec_from_file_location('_catalog_entry', p); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "x=m.build_factor_spec(); "
                "assert (x.name, x.factor)==('range_30m','range_ratio'); "
                "assert not any(name.startswith('cbond_on.domain.factors.operators') "
                "for name in sys.modules)"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
