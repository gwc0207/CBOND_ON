"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_volume_weighted_avg_size_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430VolumeWeightedAvgSizeV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_volume_weighted_avg_size_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_volume_weighted_avg_size_v1.py'
IMPLEMENTATION_SHA256 = '116b83d44ac7b0846123a204aaa36ff5cad594d3f1d1a62a2854622b891f6aff'
CONTRACT_PATH = 'operators/t1430_volume_weighted_avg_size_v1/contract.json'


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
