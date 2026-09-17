"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'stock_bond_momentum_gap_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'StockBondMomentumGapV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.stock_bond_momentum_gap_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/stock_bond_momentum_gap_v1.py'
IMPLEMENTATION_SHA256 = '0af492cf6cd4bcb3c6c5df193f87c2c19c72144112abf232432e9c8c21c5a724'
CONTRACT_PATH = 'operators/stock_bond_momentum_gap_v1/contract.json'


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
