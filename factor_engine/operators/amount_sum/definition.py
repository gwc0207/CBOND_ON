"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'amount_sum'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'AmountSumFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.amount_sum'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/amount_sum.py'
IMPLEMENTATION_SHA256 = '660cbbfb9d66aa358414a4a625c108bcc936ba9d989cd987785503bf889fdf00'
CONTRACT_PATH = 'operators/amount_sum/contract.json'


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
