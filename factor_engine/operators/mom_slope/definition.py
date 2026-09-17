"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'mom_slope'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'MomentumSlopeFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.mom_slope'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/mom_slope.py'
IMPLEMENTATION_SHA256 = 'e8dff9a4d10bba546a08483258100ef60dd361b9421f8e8d9bd329f5aa643192'
CONTRACT_PATH = 'operators/mom_slope/contract.json'


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
