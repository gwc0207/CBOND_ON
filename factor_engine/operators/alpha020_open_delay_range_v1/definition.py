"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha020_open_delay_range_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha020OpenDelayRangeV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha020_open_delay_range_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha020_open_delay_range_v1.py'
IMPLEMENTATION_SHA256 = 'e44ef2ddad357f21bb652585226906f15b683171636f713a8a61f0c39db24bfe'
CONTRACT_PATH = 'operators/alpha020_open_delay_range_v1/contract.json'


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
