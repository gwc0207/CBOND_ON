"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha012_volume_close_reversal_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha012VolumeCloseReversalV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha012_volume_close_reversal_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha012_volume_close_reversal_v1.py'
IMPLEMENTATION_SHA256 = '1d11f3f362a99085de8c3c891382f4b1f57e5427637ebdefcdf097b4b2ffbe5a'
CONTRACT_PATH = 'operators/alpha012_volume_close_reversal_v1/contract.json'


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
