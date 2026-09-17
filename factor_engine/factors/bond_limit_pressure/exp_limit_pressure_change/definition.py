"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'exp_limit_pressure_change'
FACTOR_VERSION = 'catalog-20260826-61982bae1ce3-7867b00d8d8d'
PRIMARY_FAMILY = 'bond_limit_pressure'
OPERATOR_ID = 'factor_mining_intraday_expansion_v1'
FIXED_PARAMS = {'family': 'bond_limit_pressure', 'signal': 'exp_limit_pressure_change'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/bond_limit_pressure/exp_limit_pressure_change/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_intraday_expansion_v1/contract.json'


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
