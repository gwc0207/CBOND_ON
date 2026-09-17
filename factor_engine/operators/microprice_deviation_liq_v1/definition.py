"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'microprice_deviation_liq_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'MicropriceDeviationLiqV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.microprice_deviation_liq_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/microprice_deviation_liq_v1.py'
IMPLEMENTATION_SHA256 = '372e782f7002e3a16747016f197af4e06c4f94eaec063569415a0c6fd83a049a'
CONTRACT_PATH = 'operators/microprice_deviation_liq_v1/contract.json'


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
