"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_orderbook_repricing_v1'
OPERATOR_VERSION = '20260803_orderbook_repricing_v1'
OPERATOR_CLASS = 'FactorMiningOrderbookRepricingV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_orderbook_repricing_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_orderbook_repricing_v1.py'
IMPLEMENTATION_SHA256 = 'd24776118a31287d0330007b23505d473f8985a202a09b338973356e05593621'
CONTRACT_PATH = 'operators/factor_mining_orderbook_repricing_v1/contract.json'


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
