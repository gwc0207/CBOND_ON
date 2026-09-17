"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_limit_pressure_temporal_topology_v1'
OPERATOR_VERSION = '20260803_limit_pressure_temporal_topology_v1'
OPERATOR_CLASS = 'FactorMiningLimitPressureTemporalTopologyV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_limit_pressure_temporal_topology_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_limit_pressure_temporal_topology_v1.py'
IMPLEMENTATION_SHA256 = 'c498603c4056353d94777a03871e4a643d36956aa39673e2889a795b24a3bb6c'
CONTRACT_PATH = 'operators/factor_mining_limit_pressure_temporal_topology_v1/contract.json'


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
