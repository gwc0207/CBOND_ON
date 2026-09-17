"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'icg_curve_turning_energy'
FACTOR_VERSION = 'catalog-20260826-c3de662d1c32-b43b6ca5cc80'
PRIMARY_FAMILY = 'information_clock_curve_trajectory'
OPERATOR_ID = 'factor_mining_intraday_information_clock_geometry_v1'
FIXED_PARAMS = {'family': 'information_clock_curve_trajectory', 'signal': 'icg_curve_turning_energy'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/information_clock_curve_trajectory/icg_curve_turning_energy/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_intraday_information_clock_geometry_v1/contract.json'


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
