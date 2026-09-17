from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys

from cbond_on.domain.factor_catalog import load_factor_catalog, validate_factor_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PREFIX = "cbond_on.domain.factors.operators."


def _load_definition(path: Path, factor_id: str):
    spec = spec_from_file_location(f"_factor_definition_{factor_id}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_factor_definitions_and_contracts() -> None:
    catalog = load_factor_catalog(REPO_ROOT)
    assert len(catalog) == 800

    for factor_id, row in catalog.items():
        definition = _load_definition(
            REPO_ROOT / "factor_engine" / row["definition_path"], factor_id
        )
        contract = json.loads(
            (REPO_ROOT / "factor_engine" / row["contract_path"]).read_text(encoding="utf-8")
        )
        payload = definition.definition_payload()
        spec = definition.build_factor_spec()
        binding = contract["operator_bindings"][0]

        assert contract["schema_version"] == "factor_contract/v2"
        assert payload["factor_id"] == factor_id
        assert payload["operator_id"] == row["operator_id"]
        assert payload["operator_contract_path"] == binding["operator_contract_path"]
        assert spec.name == factor_id
        assert spec.factor == row["operator_id"]
        assert spec.params == contract["fixed_params"]
        assert binding["implementation_module"].startswith(OPERATOR_PREFIX)
        assert binding["legacy_implementation_path"].startswith(
            "cbond_on/domain/factors/defs/"
        )
        assert (REPO_ROOT / "factor_engine" / binding["operator_contract_path"]).is_file()


def test_every_operator_contract_matches_runtime_source() -> None:
    catalog_path = REPO_ROOT / "factor_engine" / "catalog" / "operator_catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    operators = payload["operators"]
    assert len(operators) == 266

    for row in operators:
        implementation = REPO_ROOT / row["implementation_path"]
        contract_path = REPO_ROOT / "factor_engine" / row["operator_contract_path"]
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        assert implementation.is_file()
        assert row["implementation_module"].startswith(OPERATOR_PREFIX)
        assert contract["schema_version"] == "operator_contract/v1"
        assert contract["identity"]["operator_id"] == row["operator_id"]
        assert contract["implementation"] == {
            "module": row["implementation_module"],
            "path": row["implementation_path"],
            "sha256": row["implementation_sha256"],
            "language": "python_runtime_operator",
        }
        assert contract["migration"]["legacy_path"] == row["legacy_implementation_path"]


def test_new_operator_package_registers_exactly_the_catalog_surface_in_a_fresh_process() -> None:
    probe = """
import importlib
import json
import sys
from pathlib import Path

from cbond_on.core.registry import OperatorRegistry

root = Path('cbond_on/domain/factors/operators')
for path in sorted(root.glob('*.py')):
    if path.name == '__init__.py' or path.name.startswith('_'):
        continue
    importlib.import_module('cbond_on.domain.factors.operators.' + path.stem)
rows = {
    str(name): {
        'module': OperatorRegistry.get(name).__module__,
        'class': OperatorRegistry.get(name).__name__,
    }
    for name in sorted(OperatorRegistry.names())
}
print(json.dumps({'rows': rows, 'retired_loaded': any('.factors.defs' in name for name in sys.modules)}))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    actual = json.loads(result.stdout)
    expected = json.loads(
        (REPO_ROOT / "factor_engine" / "catalog" / "operator_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    assert actual["retired_loaded"] is False
    assert set(actual["rows"]) == {row["operator_id"] for row in expected["operators"]}
    assert all(item["module"].startswith(OPERATOR_PREFIX) for item in actual["rows"].values())


def test_catalog_validation_covers_the_full_operator_and_factor_graph() -> None:
    assert validate_factor_catalog(REPO_ROOT) == {
        "factor_count": 800,
        "operator_count": 266,
        "family_count": 174,
        "manual_override_count": 1,
        "live_release_count": 1,
        "live_released_factor_count": 50,
    }
