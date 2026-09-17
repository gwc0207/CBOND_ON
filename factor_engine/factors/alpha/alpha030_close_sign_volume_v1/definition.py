"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'alpha030_close_sign_volume_v1'
FACTOR_VERSION = 'catalog-20260826-7df991b56081-d4b87a7b6d5e'
PRIMARY_FAMILY = 'alpha'
OPERATOR_ID = 'alpha030_close_sign_volume_v1'
FIXED_PARAMS = {'delay1': 1, 'delay2': 2, 'delay3': 3, 'sum_window_long': 10, 'sum_window_short': 5}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/alpha030_close_sign_volume_v1'
CONTRACT_PATH = 'factors/alpha/alpha030_close_sign_volume_v1/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/alpha030_close_sign_volume_v1/contract.json'


def definition_payload() -> dict[str, object]:
    """Return this immutable instance's execution metadata as a fresh mapping."""

    return {
        'factor_id': FACTOR_ID,
        'factor_version': FACTOR_VERSION,
        'primary_family': PRIMARY_FAMILY,
        'operator_id': OPERATOR_ID,
        'fixed_params': dict(FIXED_PARAMS),
        'output_col': OUTPUT_COL,
        'rust_contract_id': RUST_CONTRACT_ID,
        'operator_contract_path': OPERATOR_CONTRACT_PATH,
    }


def build_factor_spec():
    """Build an exact ``FactorSpec`` without importing or registering an operator."""

    from cbond_on.domain.factors.spec import FactorSpec

    return FactorSpec(
        name=FACTOR_ID,
        factor=OPERATOR_ID,
        params=dict(FIXED_PARAMS),
        output_col=OUTPUT_COL,
        rust_contract_id=RUST_CONTRACT_ID,
    )
