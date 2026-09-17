"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha047_inverse_close_volume_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha047InverseCloseVolumeV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha047_inverse_close_volume_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha047_inverse_close_volume_v1.py'
IMPLEMENTATION_SHA256 = '6e935fef687635fe7eefe3a346e2077c75e06da4eec8ca903bad6b128e1086a2'
CONTRACT_PATH = 'operators/alpha047_inverse_close_volume_v1/contract.json'


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
