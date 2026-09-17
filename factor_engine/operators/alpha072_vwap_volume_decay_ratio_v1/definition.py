"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha072_vwap_volume_decay_ratio_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha072VwapVolumeDecayRatioV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha072_vwap_volume_decay_ratio_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha072_vwap_volume_decay_ratio_v1.py'
IMPLEMENTATION_SHA256 = 'ce3fafab9ee710f9271a8f9097555c7a054ce61ae671dca5129adf80c9d73f45'
CONTRACT_PATH = 'operators/alpha072_vwap_volume_decay_ratio_v1/contract.json'


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
