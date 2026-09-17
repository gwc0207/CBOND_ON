"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha055_close_range_volume_corr_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha055CloseRangeVolumeCorrV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha055_close_range_volume_corr_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha055_close_range_volume_corr_v1.py'
IMPLEMENTATION_SHA256 = '5d8b61ceef3d876884b574bb5550c84eb31d84d23181c8350ad6cd7e092314ec'
CONTRACT_PATH = 'operators/alpha055_close_range_volume_corr_v1/contract.json'


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
