"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_close_range_position_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430CloseRangePositionV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_close_range_position_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_close_range_position_v1.py'
IMPLEMENTATION_SHA256 = 'ea8c493528d411a7638f3fd62fafb63b7b50614e0d9c41f034ec3b7227c7dfdc'
CONTRACT_PATH = 'operators/t1430_close_range_position_v1/contract.json'


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
