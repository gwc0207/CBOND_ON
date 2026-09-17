"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'midprice_move'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'MidpriceMoveFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.midprice_move'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/midprice_move.py'
IMPLEMENTATION_SHA256 = '1709caac6b9161a8b41b1f48143869f0c80242075656daa56b23435196664da5'
CONTRACT_PATH = 'operators/midprice_move/contract.json'


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
