"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'order_flow_imbalance_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'OrderFlowImbalanceV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.order_flow_imbalance_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/order_flow_imbalance_v1.py'
IMPLEMENTATION_SHA256 = '2cc5e287bb1f105364cc5b2e4cfd084d66aea0acc30633681c84b29d4d7ce531'
CONTRACT_PATH = 'operators/order_flow_imbalance_v1/contract.json'


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
