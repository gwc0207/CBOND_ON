"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_volume_count_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430VolumeCountV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_volume_count_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_volume_count_v1.py'
IMPLEMENTATION_SHA256 = 'fbc48c88e309ed94ea15396d8ef17dfae20aee7f6c0a89353e2286949c110c7e'
CONTRACT_PATH = 'operators/t1430_volume_count_v1/contract.json'


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
