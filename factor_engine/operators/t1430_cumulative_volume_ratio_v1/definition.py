"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_cumulative_volume_ratio_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430CumulativeVolumeRatioV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_cumulative_volume_ratio_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_cumulative_volume_ratio_v1.py'
IMPLEMENTATION_SHA256 = '0f7fba518db1fd9659f2851c00f08d6575ec9d27a25eae0f8030a8c1e775274d'
CONTRACT_PATH = 'operators/t1430_cumulative_volume_ratio_v1/contract.json'


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
