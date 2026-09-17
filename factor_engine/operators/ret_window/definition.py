"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'ret_window'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'ReturnWindowFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.ret_window'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/ret_window.py'
IMPLEMENTATION_SHA256 = '8564e464b0afb2f6d8fa0458327d2194c3bdc24fa88754a4fc1ea3d6754a8a1f'
CONTRACT_PATH = 'operators/ret_window/contract.json'


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
