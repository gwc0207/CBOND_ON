from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cbond_on.common.config_utils import load_json_like
from cbond_on.infra.factors.rust_backend import RUST_FIRST_CAPABILITY_ABI


_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _ROOT / "rust" / "factor_engine" / "factor_manifest.json"
_PACK_PATH = (
    _ROOT
    / "cbond_on"
    / "config"
    / "factor"
    / "packs"
    / "live_screened_no_winsor_50_20260805.json5"
)
_PROFILE_PATH = (
    _ROOT
    / "cbond_on"
    / "factor_contracts"
    / "profiles"
    / "live50_rust50_20260806.json5"
)
_PROFILE = "live50_rust50_20260806"
_R88_PROFILE_PATH = (
    _ROOT
    / "cbond_on"
    / "factor_contracts"
    / "profiles"
    / "research_r88_rust88_20260825.json5"
)
_R88_PROFILE = "research_r88_rust88_20260825"
_CAPABILITY_ABI = "rust_factor_contracts_20260806_r1"


def _load_manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _params_sha256(params: dict) -> str:
    canonical = json.dumps(
        params,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_specs_sha256(specs: list[dict]) -> str:
    canonical = json.dumps(
        specs,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _spec_payload(raw_specs: list[dict]) -> list[dict]:
    return [
        {
            "name": str(spec["name"]),
            "factor": str(spec["factor"]),
            "params": dict(spec.get("params") or {}),
            "output_col": spec.get("output_col"),
            "rust_contract_id": spec.get("rust_contract_id"),
        }
        for spec in raw_specs
    ]


def _capability_contracts(raw_specs: list[dict]) -> list[dict]:
    contracts: list[dict] = []
    for spec in raw_specs:
        params = dict(spec.get("params") or {})
        signal = str(params.get("signal", "")).strip() or None
        contracts.append(
            {
                "id": str(spec["rust_contract_id"]),
                "output_col": str(spec.get("output_col") or spec["name"]),
                "factor": str(spec["factor"]),
                "signal": signal,
                "params_sha256": _params_sha256(params),
            }
        )
    return contracts


def test_rust_manifest_declares_loaded_capability_as_runtime_authority() -> None:
    manifest = _load_manifest()

    assert RUST_FIRST_CAPABILITY_ABI == _CAPABILITY_ABI
    assert manifest["schema_version"] == 2
    assert manifest["module"] == "cbond_on_rust"
    assert manifest["runtime_authority"] == {
        "kind": "loaded_extension_capability",
        "function": "cbond_on_rust.factor_capabilities",
        "capability_field": "factor_contracts",
        "capability_abi_revision": _CAPABILITY_ABI,
        "static_manifest_role": (
            "source_and_contract_documentation_only; runtime executability is decided "
            "solely by the loaded extension capability payload"
        ),
    }

    catalog = manifest["factors"]
    assert manifest["total_factors"] == len(catalog)
    assert len({item["factor"] for item in catalog}) == len(catalog)
    assert {item["rust_status"] for item in catalog} <= {
        "implemented",
        "implemented_frozen_contracts_only",
        "pending",
    }
    for item in catalog:
        source = _ROOT / "cbond_on" / "domain" / "factors" / "operators" / item["python_file"]
        assert source.is_file(), item


def test_frozen_live50_contract_cannot_drift_from_pack_or_profile() -> None:
    manifest = _load_manifest()
    pack = load_json_like(_PACK_PATH)
    profile = load_json_like(_PROFILE_PATH)
    raw_specs = list(pack["factors"])

    frozen = [item for item in manifest["frozen_contracts"] if item["profile"] == _PROFILE]
    assert len(frozen) == 1
    frozen_contract = frozen[0]
    expected_specs = _spec_payload(raw_specs)
    expected_contracts = _capability_contracts(raw_specs)

    assert len(raw_specs) == 50
    assert frozen_contract["execution_policy"] == "rust_first"
    assert frozen_contract["pack_path"] == "cbond_on/config/factor/packs/live_screened_no_winsor_50_20260805.json5"
    assert frozen_contract["profile_path"] == "cbond_on/factor_contracts/profiles/live50_rust50_20260806.json5"
    assert frozen_contract["contract_count"] == 50
    assert frozen_contract["specs_sha256"] == _canonical_specs_sha256(expected_specs)
    assert profile["admission_profile"] == _PROFILE
    assert profile["execution_policy"] == "rust_first"
    assert profile["specs_sha256"] == frozen_contract["specs_sha256"]
    assert profile["factors"] == [item["output_col"] for item in expected_contracts]
    assert frozen_contract["contracts"] == expected_contracts

    ids = [item["id"] for item in expected_contracts]
    outputs = [item["output_col"] for item in expected_contracts]
    assert len(set(ids)) == len(ids) == 50
    assert len(set(outputs)) == len(outputs) == 50
    assert all(item["id"].startswith("live50_r5/") for item in expected_contracts)
    assert all(len(item["params_sha256"]) == 64 for item in expected_contracts)

    catalog = {item["factor"]: item for item in manifest["factors"]}
    for factor in {item["factor"] for item in expected_contracts}:
        entry = catalog[factor]
        assert entry["rust_status"] in {
            "implemented",
            "implemented_frozen_contracts_only",
        }
        if entry["rust_status"] == "implemented_frozen_contracts_only":
            assert entry["contract_scope"] == _PROFILE


def test_r88_complete_research_contract_scope_is_exact() -> None:
    manifest = _load_manifest()
    profile = load_json_like(_R88_PROFILE_PATH)
    raw_specs = list(profile["factor_specs"])
    admitted_specs = [
        spec
        for spec in raw_specs
        if str(spec.get("rust_contract_id") or "").startswith("research_r88_20260825/")
    ]
    expected_contracts = _capability_contracts(admitted_specs)
    scopes = [
        item
        for item in manifest["research_contracts"]
        if item["profile"] == _R88_PROFILE
    ]
    assert len(scopes) == 1
    scope = scopes[0]
    assert scope["research_only"] is True
    assert scope["execution_policy"] == "rust_first"
    assert scope["profile_path"] == (
        "cbond_on/factor_contracts/profiles/research_r88_rust88_20260825.json5"
    )
    assert scope["execution_status"] == "eligible_for_fresh_rust_backfill"
    assert scope["model_training_ready"] is False
    assert scope["factor_count"] == len(raw_specs) == 88
    assert scope["inherited_live50_contract_count"] == 50
    assert scope["research_contract_count"] == len(admitted_specs) == 38
    assert scope["contract_id_namespace"] == "research_r88_20260825/<signal>/v1"
    assert scope["specs_sha256"] == profile["specs_sha256"]
    assert profile["pending_rust_contract_factors"] == []
    assert scope["contracts"] == expected_contracts

    catalog = {item["factor"]: item for item in manifest["factors"]}
    assert "research_partial_contracts" not in manifest
    # R88 inherits the ordinary live50 generic implementations as well as
    # research-only exact instances.  Only the latter must carry the R88
    # contract scope; forcing generic legacy kernels to become
    # frozen-contract-only would falsely narrow their pre-existing capability.
    for factor in {item["factor"] for item in raw_specs}:
        assert catalog[factor]["rust_status"] in {
            "implemented",
            "implemented_frozen_contracts_only",
        }
    for factor in {item["factor"] for item in admitted_specs}:
        entry = catalog[factor]
        assert entry["rust_status"] == "implemented_frozen_contracts_only"
        scopes = {entry["contract_scope"], *entry.get("additional_contract_scopes", [])}
        assert _R88_PROFILE in scopes
