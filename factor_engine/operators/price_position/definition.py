"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'price_position'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'PricePositionFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.price_position'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/price_position.py'
IMPLEMENTATION_SHA256 = '711ded0fd9a4808e6152f764a5b0340bf302bf850971e75bd6e7470c9df5e8b6'
CONTRACT_PATH = 'operators/price_position/contract.json'


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
