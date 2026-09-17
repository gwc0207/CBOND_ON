"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_order_flow_imbalance_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430OrderFlowImbalanceV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_order_flow_imbalance_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_order_flow_imbalance_v1.py'
IMPLEMENTATION_SHA256 = '9fd92905d62baf885d25bebb1235cfc6a1296d494f89e49d71a0d34ed7898cbc'
CONTRACT_PATH = 'operators/t1430_order_flow_imbalance_v1/contract.json'


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
