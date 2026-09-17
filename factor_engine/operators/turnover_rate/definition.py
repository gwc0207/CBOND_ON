"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'turnover_rate'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'TurnoverRateFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.turnover_rate'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/turnover_rate.py'
IMPLEMENTATION_SHA256 = '6babbefa1113335ee1fba79d090b54410a1d6abb8ba72cc925fba2f16ef090f2'
CONTRACT_PATH = 'operators/turnover_rate/contract.json'


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
