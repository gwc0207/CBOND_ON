"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha025_return_volume_vwap_range_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha025ReturnVolumeVwapRangeV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha025_return_volume_vwap_range_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha025_return_volume_vwap_range_v1.py'
IMPLEMENTATION_SHA256 = '74560f900ef6bac1591599d9d70edf92a1b9de65df7f8bfb128fa8129c807125'
CONTRACT_PATH = 'operators/alpha025_return_volume_vwap_range_v1/contract.json'


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
