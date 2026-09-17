"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha053_price_position_delta_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha053PricePositionDeltaV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha053_price_position_delta_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha053_price_position_delta_v1.py'
IMPLEMENTATION_SHA256 = '1b0340dc3fd1d213b8cc37bc43f7a6117e790c9b510349ba49fd27262177c527'
CONTRACT_PATH = 'operators/alpha053_price_position_delta_v1/contract.json'


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
