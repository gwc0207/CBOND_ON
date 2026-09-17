"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha002_corr_volume_return_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha002CorrVolumeReturnV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha002_corr_volume_return_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha002_corr_volume_return_v1.py'
IMPLEMENTATION_SHA256 = 'c27a29a8e6121ef4e72b89cc1f462f96010f5f6c199cdbc0d3cb8aee7daf93c1'
CONTRACT_PATH = 'operators/alpha002_corr_volume_return_v1/contract.json'


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
