"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_incremental_flow_geometry_v1'
OPERATOR_VERSION = '20260804_incremental_flow_geometry_v1'
OPERATOR_CLASS = 'FactorMiningIncrementalFlowGeometryV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_incremental_flow_geometry_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_incremental_flow_geometry_v1.py'
IMPLEMENTATION_SHA256 = '3d7891c8013115a3e70d39cf9e0ca13f5fd2b6bb28df5aa86b6f24322cbf91f6'
CONTRACT_PATH = 'operators/factor_mining_incremental_flow_geometry_v1/contract.json'


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
