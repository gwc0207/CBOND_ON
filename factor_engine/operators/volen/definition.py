"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'volen'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'VolenFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.volen'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/volen.py'
IMPLEMENTATION_SHA256 = 'e40e799d1756068a6fea5358a058cbfe27347d9e3761b39c8188b47956ee3545'
CONTRACT_PATH = 'operators/volen/contract.json'


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
