"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha013_cov_close_volume_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha013CovCloseVolumeV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha013_cov_close_volume_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha013_cov_close_volume_v1.py'
IMPLEMENTATION_SHA256 = '87e0a315b096aff0f62090553596507280cb5e2c0cb420ba71063703b8906b96'
CONTRACT_PATH = 'operators/alpha013_cov_close_volume_v1/contract.json'


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
