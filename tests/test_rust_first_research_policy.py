from __future__ import annotations

import pytest

from cbond_on.infra.factors.quality import load_factor_specs_from_cfg
from cbond_on.common.factor_execution_policy import (
    validate_factor_execution_policy,
)
from cbond_on.workflows.research.factor_batch import validate_research_execution_policy


def _research_cfg(*, compute: dict, **extra: object) -> dict:
    return {
        "research_only": True,
        "compute": compute,
        "factors": [],
        "factor_files": [],
        **extra,
    }


def test_new_research_batch_requires_explicit_rust_first_policy() -> None:
    with pytest.raises(ValueError, match="must use compute.execution_policy='rust_first'"):
        validate_research_execution_policy(
            _research_cfg(compute={"engine": "rust"})
        )


def test_rust_first_research_batch_requires_public_rust_engine() -> None:
    with pytest.raises(ValueError, match="compute.engine='rust'"):
        validate_research_execution_policy(
            _research_cfg(
                compute={"engine": "rust_python_hybrid", "execution_policy": "rust_first"}
            )
        )


def test_rust_first_research_batch_is_accepted() -> None:
    validate_research_execution_policy(
        _research_cfg(compute={"engine": "rust", "execution_policy": "rust_first"})
    )


def test_archival_python_reproduction_is_not_an_executable_research_exception() -> None:
    cfg = _research_cfg(
        compute={"engine": "python", "execution_policy": "legacy_reference_only"},
        legacy_reference_only=True,
        legacy_reference_reason="reproduce a pre-rust archived report",
    )
    with pytest.raises(ValueError, match="execution_policy='rust_first'"):
        validate_research_execution_policy(cfg)


def test_nonresearch_batch_is_also_rust_first() -> None:
    with pytest.raises(ValueError, match="compute.engine='rust'"):
        validate_factor_execution_policy(
            {
                "compute": {"engine": "python", "execution_policy": "rust_first"},
                "factors": [],
                "factor_files": [],
            },
            scope="factor_batch",
        )


def test_factor_spec_loader_retains_exact_rust_contract_id() -> None:
    specs = load_factor_specs_from_cfg(
        {
            "factors": [
                {
                    "name": "candidate_signal",
                    "factor": "candidate_factor_v1",
                    "params": {"window": 20},
                    "rust_contract_id": "research/candidate_signal/v1",
                }
            ],
            "factor_files": [],
        }
    )
    assert specs[0].rust_contract_id == "research/candidate_signal/v1"
