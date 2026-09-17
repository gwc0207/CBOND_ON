"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'vwap_gap'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'VwapGapFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.vwap_gap'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/vwap_gap.py'
IMPLEMENTATION_SHA256 = 'c50a2637e0bc424839191437d81d95eb97b25a312b6a4b447850363aad0b77a6'
CONTRACT_PATH = 'operators/vwap_gap/contract.json'


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
