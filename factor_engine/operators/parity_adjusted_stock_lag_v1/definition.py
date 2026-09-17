"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'parity_adjusted_stock_lag_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'ParityAdjustedStockLagV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.parity_adjusted_stock_lag_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/parity_adjusted_stock_lag_v1.py'
IMPLEMENTATION_SHA256 = '515e887d710990d889a7dd31fa27d44bd1f7efc741fed0bee608b662055abcc1'
CONTRACT_PATH = 'operators/parity_adjusted_stock_lag_v1/contract.json'


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
