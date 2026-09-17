"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'ret_open_to_time'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'ReturnOpenToTimeFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.ret_open_to_time'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/ret_open_to_time.py'
IMPLEMENTATION_SHA256 = 'd86cadd0a0a9768245480381df5e7cd1ec4a6a106028ecfe1fce65184d380b88'
CONTRACT_PATH = 'operators/ret_open_to_time/contract.json'


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
