"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_cross_asset_event_sequence_topology_v1'
OPERATOR_VERSION = '20260803_cross_asset_event_sequence_topology_v1'
OPERATOR_CLASS = 'FactorMiningCrossAssetEventSequenceTopologyV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_cross_asset_event_sequence_topology_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_cross_asset_event_sequence_topology_v1.py'
IMPLEMENTATION_SHA256 = '588bb615960806bf85846a20e8960264c854d17fec4d79b733c00ebf59b19cdd'
CONTRACT_PATH = 'operators/factor_mining_cross_asset_event_sequence_topology_v1/contract.json'


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
