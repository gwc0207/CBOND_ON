"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'trade_intensity_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'TradeIntensityV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.trade_intensity_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/trade_intensity_v1.py'
IMPLEMENTATION_SHA256 = '130950619e3170bfa820b4b3b283d308eb13714f50a13f6c04e9e7525f56a0e5'
CONTRACT_PATH = 'operators/trade_intensity_v1/contract.json'


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
