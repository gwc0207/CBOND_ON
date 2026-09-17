"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_volume_min_v2'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430VolumeMinV2'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_volume_min_v2'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_volume_min_v2.py'
IMPLEMENTATION_SHA256 = '458994a6fa24c837f8910bb8955646251f9e5106dd01e153fcdfb9b66e08417a'
CONTRACT_PATH = 'operators/t1430_volume_min_v2/contract.json'


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
