"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'microprice_deviation_liq_cond_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'MicropriceDeviationLiqCondV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.microprice_deviation_liq_cond_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/microprice_deviation_liq_cond_v1.py'
IMPLEMENTATION_SHA256 = '838bedb4d32058894785b4bd7ec86fc377a9379adecb769f9b1b3e94ccb9f771'
CONTRACT_PATH = 'operators/microprice_deviation_liq_cond_v1/contract.json'


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
