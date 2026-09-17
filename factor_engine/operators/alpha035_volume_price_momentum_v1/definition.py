"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha035_volume_price_momentum_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha035VolumePriceMomentumV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha035_volume_price_momentum_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha035_volume_price_momentum_v1.py'
IMPLEMENTATION_SHA256 = 'e9542bc101ec77c0a2658ab6a7242407f7cd8c8a3e9d26b5c50c2c2fd0c7c587'
CONTRACT_PATH = 'operators/alpha035_volume_price_momentum_v1/contract.json'


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
