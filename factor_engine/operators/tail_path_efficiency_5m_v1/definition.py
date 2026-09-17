"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'tail_path_efficiency_5m_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'TailPathEfficiency5mV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.tail_path_efficiency_5m_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/tail_path_efficiency_5m_v1.py'
IMPLEMENTATION_SHA256 = '149579354bd5359250ade5ecb303164c63a66ff4c23b85829c2f62b93de9e618'
CONTRACT_PATH = 'operators/tail_path_efficiency_5m_v1/contract.json'


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
