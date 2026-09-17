"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'vwap'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'VwapFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.vwap'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/vwap.py'
IMPLEMENTATION_SHA256 = 'f7eff67110ac41434436b673c4f07361fcd9dbb92e176e7eea7c01b3c407df47'
CONTRACT_PATH = 'operators/vwap/contract.json'


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
