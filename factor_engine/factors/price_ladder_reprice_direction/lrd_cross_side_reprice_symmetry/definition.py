"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'lrd_cross_side_reprice_symmetry'
FACTOR_VERSION = 'catalog-20260826-d24776118a31-20f9f38e3c19'
PRIMARY_FAMILY = 'price_ladder_reprice_direction'
OPERATOR_ID = 'factor_mining_orderbook_repricing_v1'
FIXED_PARAMS = {'family': 'price_ladder_reprice_direction', 'signal': 'lrd_cross_side_reprice_symmetry'}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/lrd_cross_side_reprice_symmetry'
CONTRACT_PATH = 'factors/price_ladder_reprice_direction/lrd_cross_side_reprice_symmetry/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_orderbook_repricing_v1/contract.json'


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
