"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'volume_imbalance'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'VolumeImbalanceFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.volume_imbalance'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/volume_imbalance.py'
IMPLEMENTATION_SHA256 = 'e94fb77e2a1d58276e8bfd0fef65ab7cfb7698a3693089dd036ad22264fe861c'
CONTRACT_PATH = 'operators/volume_imbalance/contract.json'


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
