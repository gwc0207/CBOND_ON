"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_intraday_trade_grid_topology_v1'
OPERATOR_VERSION = '20260803_intraday_trade_grid_topology_v1'
OPERATOR_CLASS = 'FactorMiningIntradayTradeGridTopologyV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_intraday_trade_grid_topology_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_intraday_trade_grid_topology_v1.py'
IMPLEMENTATION_SHA256 = '57fe2f067385a146828774652f75d4da80a09cac5af62cebf6cbc94dbb580bcb'
CONTRACT_PATH = 'operators/factor_mining_intraday_trade_grid_topology_v1/contract.json'


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
