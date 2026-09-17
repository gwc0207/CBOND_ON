"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'amihud_illiq'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'AmihudIlliqFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.amihud_illiq'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/amihud_illiq.py'
IMPLEMENTATION_SHA256 = 'ae3f8e37e85a0c62973fbec8d0126ddcbbafe7825267dd05d17268b037a7c234'
CONTRACT_PATH = 'operators/amihud_illiq/contract.json'


def definition_payload() -> dict[str, object]:
    """Return immutable identity metadata without importing the runtime implementation."""

    return {
        'operator_id': OPERATOR_ID,
        'operator_version': OPERATOR_VERSION,
        'operator_class': OPERATOR_CLASS,
        'implementation_module': IMPLEMENTATION_MODULE,
        'implementation_path': IMPLEMENTATION_PATH,
        'implementation_sha256': IMPLEMENTATION_SHA256,
        'contract_path': CONTRACT_PATH,
    }
