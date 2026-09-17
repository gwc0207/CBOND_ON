"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'ai_factory_wave80_intraday_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'AiFactoryWave80IntradayV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.ai_factory_wave80_intraday_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/ai_factory_wave80_intraday_v1.py'
IMPLEMENTATION_SHA256 = 'cd6819391e6f1ddf3cd85ca4e7b25048bb2428295c6e819303ec38eeba35bf2a'
CONTRACT_PATH = 'operators/ai_factory_wave80_intraday_v1/contract.json'


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
