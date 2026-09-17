"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'range_ratio'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'RangeRatioFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.range_ratio'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/range_ratio.py'
IMPLEMENTATION_SHA256 = 'a7905bcb2b23ac965257c01c138b6193b8e0901a2f8cd8f221180656021784c8'
CONTRACT_PATH = 'operators/range_ratio/contract.json'


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
