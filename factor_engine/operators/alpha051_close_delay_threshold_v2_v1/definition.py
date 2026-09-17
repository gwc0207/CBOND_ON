"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha051_close_delay_threshold_v2_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha051CloseDelayThresholdV2V1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha051_close_delay_threshold_v2_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha051_close_delay_threshold_v2_v1.py'
IMPLEMENTATION_SHA256 = '2b73ba2e14e7851f83bcfe00de443316113dfa1e5e147e114943da594ce70e97'
CONTRACT_PATH = 'operators/alpha051_close_delay_threshold_v2_v1/contract.json'


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
