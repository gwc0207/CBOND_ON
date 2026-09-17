"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_volume_count_v3'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430VolumeCountV3'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_volume_count_v3'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_volume_count_v3.py'
IMPLEMENTATION_SHA256 = '945f9c9223a2b69b14e4e13e943f6fd030286bb64b9461cf38be62bb27a4e7b2'
CONTRACT_PATH = 'operators/t1430_volume_count_v3/contract.json'


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
