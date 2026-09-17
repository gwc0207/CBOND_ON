"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha014_return_open_volume_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha014ReturnOpenVolumeV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha014_return_open_volume_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha014_return_open_volume_v1.py'
IMPLEMENTATION_SHA256 = 'e28e9e58124109e16744746021a77903e1fbfaf2a28bbca0c51aa08e35bbc807'
CONTRACT_PATH = 'operators/alpha014_return_open_volume_v1/contract.json'


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
