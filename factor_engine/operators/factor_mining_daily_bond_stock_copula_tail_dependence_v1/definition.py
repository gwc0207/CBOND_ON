"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_daily_bond_stock_copula_tail_dependence_v1'
OPERATOR_VERSION = '20260803_daily_bond_stock_copula_tail_dependence_v1'
OPERATOR_CLASS = 'FactorMiningDailyBondStockCopulaTailDependenceV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_daily_bond_stock_copula_tail_dependence_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_daily_bond_stock_copula_tail_dependence_v1.py'
IMPLEMENTATION_SHA256 = '6c0e7d7d63f68a6692f32ba86a94eeca2045125cfe7d813d148851162ed41b44'
CONTRACT_PATH = 'operators/factor_mining_daily_bond_stock_copula_tail_dependence_v1/contract.json'


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
