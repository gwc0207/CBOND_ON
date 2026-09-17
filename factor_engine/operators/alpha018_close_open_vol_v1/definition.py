"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha018_close_open_vol_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha018CloseOpenVolV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha018_close_open_vol_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha018_close_open_vol_v1.py'
IMPLEMENTATION_SHA256 = 'a4111f740a691f13ea3f40326912604bd4c3021e43f589cc366faf571915f15b'
CONTRACT_PATH = 'operators/alpha018_close_open_vol_v1/contract.json'


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
