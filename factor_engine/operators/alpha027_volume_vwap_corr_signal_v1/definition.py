"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha027_volume_vwap_corr_signal_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha027VolumeVwapCorrSignalV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha027_volume_vwap_corr_signal_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha027_volume_vwap_corr_signal_v1.py'
IMPLEMENTATION_SHA256 = 'a9073dfbf47b68a07a2ca140775e107b6e13ea7746cb4a55f74a99927fe30c0a'
CONTRACT_PATH = 'operators/alpha027_volume_vwap_corr_signal_v1/contract.json'


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
