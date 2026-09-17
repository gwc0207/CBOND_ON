"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'spread'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'SpreadFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.spread'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/spread.py'
IMPLEMENTATION_SHA256 = '79130d541b49d72ba7669d93f6dccbc56d745d99a652e3e10e8a2698ea2bbbb8'
CONTRACT_PATH = 'operators/spread/contract.json'


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
