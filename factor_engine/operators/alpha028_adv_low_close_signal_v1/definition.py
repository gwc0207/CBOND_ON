"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha028_adv_low_close_signal_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha028AdvLowCloseSignalV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha028_adv_low_close_signal_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha028_adv_low_close_signal_v1.py'
IMPLEMENTATION_SHA256 = '968a61424a16243ddb7cedb39e75ad7473ccb45f6da00af9f5a9b90960ec88ce'
CONTRACT_PATH = 'operators/alpha028_adv_low_close_signal_v1/contract.json'


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
