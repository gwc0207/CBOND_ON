from __future__ import annotations

from pathlib import Path

import cbond_on.app.usecases.ai_factor_factory as factory
import pytest
from cbond_on.app.usecases.ai_factor_factory import (
    FactorCandidateDraft,
    candidate_output_schema,
    review_candidate,
    write_candidate_package,
)


def _rust_code(*, body: str) -> str:
    return f"""
pub fn compute_candidate(last: f64, volume: f64) -> Option<f64> {{
    if last <= 0.0 || volume <= 0.0 {{
        return None;
    }}
    {body}
}}
"""


def _candidate(
    *,
    key: str,
    formula: str,
    rust_body: str,
    fields: list[str],
    contract_id: str | None = None,
) -> FactorCandidateDraft:
    return FactorCandidateDraft(
        factor_key=key,
        factor_name=key,
        formula=formula,
        rationale="test candidate",
        rust_code=_rust_code(body=rust_body),
        config_spec={
            "name": key,
            "factor": "rust_factory_test_kernel_v1",
            "params": {"window_minutes": 30},
            **({"rust_contract_id": contract_id} if contract_id is not None else {}),
        },
        used_panel_fields=fields,
        time_visibility="only uses the T 14:30 panel",
    )


def _error_codes(candidate: FactorCandidateDraft) -> set[str]:
    return {finding.code for finding in review_candidate(candidate) if finding.severity == "error"}


def test_review_rejects_existing_factor_key() -> None:
    candidate = _candidate(
        key="t1430_w80_ret_depth_liq_10m_l1_v1",
        formula="a distinct Rust candidate using volume",
        rust_body="let window = last * volume; Some(window)",
        fields=["trade_time", "volume", "last"],
        contract_id="research/t1430_w80_ret_depth_liq_10m_l1_v1/v1",
    )

    assert "existing_factor_key" in _error_codes(candidate)


def test_review_rejects_duplicate_volume_formula_family_with_rust_draft() -> None:
    candidate = _candidate(
        key="t1430_new_volume_peak_rust_v1",
        formula="max(volume) over a 30 minute window",
        rust_body="let max_volume = volume.max(last); Some(max_volume)",
        fields=["trade_time", "volume"],
        contract_id="research/t1430_new_volume_peak_rust_v1/v1",
    )

    assert "duplicate_formula_family" in _error_codes(candidate)


def test_review_allows_cross_field_interaction_using_volume_component() -> None:
    candidate = _candidate(
        key="t1430_new_price_volume_interaction_rust_v1",
        formula="price-volume interaction ratio over a 30 minute window",
        rust_body="let signal = volume / last; Some(signal)",
        fields=["trade_time", "volume", "last"],
        contract_id="research/t1430_new_price_volume_interaction_rust_v1/v1",
    )

    assert "duplicate_formula_family" not in _error_codes(candidate)


def test_legacy_python_payload_is_read_for_diagnostics_but_rejected() -> None:
    payload = {
        "factor_key": "t1430_legacy_python_payload_v1",
        "factor_name": "t1430_legacy_python_payload_v1",
        "formula": "legacy formula",
        "rationale": "legacy candidate",
        "python_code": "print('not executable')",
        "config_spec": {
            "name": "t1430_legacy_python_payload_v1",
            "factor": "rust_factory_test_kernel_v1",
            "params": {},
            "rust_contract_id": "research/t1430_legacy_python_payload_v1/v1",
        },
        "time_visibility": "T 14:30 only",
    }

    candidate = FactorCandidateDraft.from_payload(payload)
    assert candidate.legacy_python_code == "print('not executable')"
    assert {"legacy_python_code", "rust_code"}.issubset(_error_codes(candidate))
    serialized = candidate.to_payload()
    assert "python_code" not in serialized
    assert serialized["legacy_python_code"] == "print('not executable')"


def test_review_requires_exact_rust_contract_id() -> None:
    candidate = _candidate(
        key="t1430_missing_rust_contract_v1",
        formula="last marker",
        rust_body="Some(last)",
        fields=["trade_time", "last"],
    )

    assert "rust_contract_id" in _error_codes(candidate)


def test_review_rejects_rust_direct_io() -> None:
    candidate = _candidate(
        key="t1430_rust_io_rejected_v1",
        formula="last marker",
        rust_body='let _text = std::fs::read_to_string("forbidden"); Some(last)',
        fields=["trade_time", "last"],
        contract_id="research/t1430_rust_io_rejected_v1/v1",
    )

    assert "rust_direct_io" in _error_codes(candidate)


def test_family_level_dify_output_fails_closed_before_candidate_staging() -> None:
    response = {
        "data": {
            "outputs": {
                "family_json": '{"factor_families":[{"family_name":"safe_family"}]}'
            }
        }
    }

    with pytest.raises(ValueError, match="not executable candidates"):
        factory._extract_candidates_from_dify_response(response)


def test_candidate_schema_and_package_are_rust_only(tmp_path: Path, monkeypatch) -> None:
    schema_candidate = candidate_output_schema()["candidates"][0]
    assert "rust_code" in schema_candidate
    assert "python_code" not in schema_candidate
    assert schema_candidate["config_spec"]["rust_contract_id"]

    candidate = _candidate(
        key="t1430_rust_package_only_v1",
        formula="last marker over a 30 minute window",
        rust_body="let window = last * volume; Some(window)",
        fields=["trade_time", "last", "volume"],
        contract_id="research/t1430_rust_package_only_v1/v1",
    )
    findings = review_candidate(candidate)
    assert not [finding for finding in findings if finding.severity == "error"]
    monkeypatch.setattr(factory, "_candidate_root", lambda: tmp_path)

    root = write_candidate_package(candidate, findings)

    assert (root / "t1430_rust_package_only_v1.rs.draft").read_text(encoding="utf-8") == candidate.rust_code
    assert not (root / "t1430_rust_package_only_v1.py.draft").exists()
    requirement = (root / "rust_contract_requirement.json").read_text(encoding="utf-8")
    assert '"execution_policy": "rust_first"' in requirement
    assert '"compiled_capability_proof_required": true' in requirement
    static_review = (root / "static_review.json").read_text(encoding="utf-8")
    assert '"research_batch_permitted": false' in static_review
