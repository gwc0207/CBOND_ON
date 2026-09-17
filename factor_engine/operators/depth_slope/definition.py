"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'depth_slope'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'DepthSlopeFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.depth_slope'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/depth_slope.py'
IMPLEMENTATION_SHA256 = '42d3b7cdac9bb6e819e18e228fce178841130e7868225d43b197ff08d9c61126'
CONTRACT_PATH = 'operators/depth_slope/contract.json'


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
