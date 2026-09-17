"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha041_geometric_mean_vwap_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha041GeometricMeanVwapV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha041_geometric_mean_vwap_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha041_geometric_mean_vwap_v1.py'
IMPLEMENTATION_SHA256 = '090c2e8f9f160faa3a30e47e06f48e6ee99c201688766f330a941144c43ff3d4'
CONTRACT_PATH = 'operators/alpha041_geometric_mean_vwap_v1/contract.json'


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
