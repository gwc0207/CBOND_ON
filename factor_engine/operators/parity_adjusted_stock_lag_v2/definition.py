"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'parity_adjusted_stock_lag_v2'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'ParityAdjustedStockLagV2Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.parity_adjusted_stock_lag_v2'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/parity_adjusted_stock_lag_v2.py'
IMPLEMENTATION_SHA256 = '9c61e03264440970c120085bce74e2f9cfc16b8d9d31133dfd273e8e66af7d80'
CONTRACT_PATH = 'operators/parity_adjusted_stock_lag_v2/contract.json'


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
