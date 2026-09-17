"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'mom_slope_30m'
FACTOR_VERSION = 'catalog-20260826-e8dff9a4d10b-9ee4e7fefde7'
PRIMARY_FAMILY = 'momentum'
OPERATOR_ID = 'mom_slope'
FIXED_PARAMS = {'price_col': 'last', 'window_minutes': 30}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/mom_slope_30m'
CONTRACT_PATH = 'factors/momentum/mom_slope_30m/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/mom_slope/contract.json'


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
