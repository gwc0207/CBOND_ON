"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'price_level_position_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'PriceLevelPositionV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.price_level_position_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/price_level_position_v1.py'
IMPLEMENTATION_SHA256 = '7b0612937fd4c61ff9ee4835869be6238623d071d2ff946cccaecfdcf9e4372f'
CONTRACT_PATH = 'operators/price_level_position_v1/contract.json'


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
