"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_intraday_state_gated_microstructure_v1'
OPERATOR_VERSION = '20260803_intraday_state_gated_microstructure_v1'
OPERATOR_CLASS = 'FactorMiningIntradayStateGatedMicrostructureV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_intraday_state_gated_microstructure_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_intraday_state_gated_microstructure_v1.py'
IMPLEMENTATION_SHA256 = '3bfbf18c9c0ca52e073259aac040fc2f51f3f903fc94adce3f4a26ef4d17d7b9'
CONTRACT_PATH = 'operators/factor_mining_intraday_state_gated_microstructure_v1/contract.json'


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
