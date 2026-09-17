"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'lrd_price_gap_absorption_efficiency'
FACTOR_VERSION = 'catalog-20260826-7c048e774a48-ceda112ac51f'
PRIMARY_FAMILY = 'lunch_price_discontinuity_resolution'
OPERATOR_ID = 'factor_mining_intraday_lunch_reopen_dynamics_v1'
FIXED_PARAMS = {'family': 'lunch_price_discontinuity_resolution', 'signal': 'lrd_price_gap_absorption_efficiency'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/lunch_price_discontinuity_resolution/lrd_price_gap_absorption_efficiency/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_intraday_lunch_reopen_dynamics_v1/contract.json'


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
