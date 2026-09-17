"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha032_vwap_close_mean_reversion_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha032VwapCloseMeanReversionV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha032_vwap_close_mean_reversion_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha032_vwap_close_mean_reversion_v1.py'
IMPLEMENTATION_SHA256 = 'edf568de95f103e311bbddd2f76a5150369e12b0265991ca0f17a3557829f577'
CONTRACT_PATH = 'operators/alpha032_vwap_close_mean_reversion_v1/contract.json'


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
