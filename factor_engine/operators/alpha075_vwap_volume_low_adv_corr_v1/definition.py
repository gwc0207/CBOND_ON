"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha075_vwap_volume_low_adv_corr_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha075VwapVolumeLowAdvCorrV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha075_vwap_volume_low_adv_corr_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha075_vwap_volume_low_adv_corr_v1.py'
IMPLEMENTATION_SHA256 = '1ed58065daf91fda47c2db4145e71fc85068d0717eb3e5516f0a16fb57f4900e'
CONTRACT_PATH = 'operators/alpha075_vwap_volume_low_adv_corr_v1/contract.json'


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
