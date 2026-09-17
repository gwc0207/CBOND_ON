"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'depth_imbalance'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'DepthImbalanceFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.depth_imbalance'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/depth_imbalance.py'
IMPLEMENTATION_SHA256 = '569037f650bc0e562520a3973cb5cc23f51ef7349b168cc101cfb77e8b93fcdb'
CONTRACT_PATH = 'operators/depth_imbalance/contract.json'


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
