"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_amount_accel_depth_delta_v2'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430AmountAccelDepthDeltaV2'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_amount_accel_depth_delta_v2'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_amount_accel_depth_delta_v2.py'
IMPLEMENTATION_SHA256 = '00b5ae320b0c4b1524cb86af84d7e6713328aac12b6701ed6772786c9e5f0869'
CONTRACT_PATH = 'operators/t1430_amount_accel_depth_delta_v2/contract.json'


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
