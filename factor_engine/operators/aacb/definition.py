"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'aacb'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'AacbFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.aacb'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/aacb.py'
IMPLEMENTATION_SHA256 = 'ac596a0775721dedcaa7d1c4e981172a464899f268b6f4ace3b2079755b29a93'
CONTRACT_PATH = 'operators/aacb/contract.json'


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
