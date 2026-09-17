"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'return_skew'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'ReturnSkewFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.return_skew'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/return_skew.py'
IMPLEMENTATION_SHA256 = '781347e767ae1e33c934b91d120638330f3c11d6730afdbbbbc79d76e8403b56'
CONTRACT_PATH = 'operators/return_skew/contract.json'


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
