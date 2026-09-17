"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_intraday_microstructure_v2'
OPERATOR_VERSION = '20260803_microstructure_v2'
OPERATOR_CLASS = 'FactorMiningIntradayMicrostructureV2'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_intraday_microstructure_v2'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_intraday_microstructure_v2.py'
IMPLEMENTATION_SHA256 = 'c00f75da43af637c0c3b9b733d0f2d96bed92cc60bb971e60e06b17e04e1f60e'
CONTRACT_PATH = 'operators/factor_mining_intraday_microstructure_v2/contract.json'


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
