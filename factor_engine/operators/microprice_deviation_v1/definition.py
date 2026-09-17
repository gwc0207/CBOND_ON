"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'microprice_deviation_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'MicropriceDeviationV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.microprice_deviation_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/microprice_deviation_v1.py'
IMPLEMENTATION_SHA256 = '4282c3c800eb74e1b7d1e0ff9d018e560cbd7e229adef97652c6889221050d0d'
CONTRACT_PATH = 'operators/microprice_deviation_v1/contract.json'


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
