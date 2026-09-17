"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_spread_width_norm_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430SpreadWidthNormV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_spread_width_norm_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_spread_width_norm_v1.py'
IMPLEMENTATION_SHA256 = '521bd0b8cc553d9132e2d9b185cdc7f9d4ae7418853d26aeae06ee294336b075'
CONTRACT_PATH = 'operators/t1430_spread_width_norm_v1/contract.json'


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
