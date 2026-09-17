"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'csn_taq_buy_initiation_neighbor_gap'
FACTOR_VERSION = 'catalog-20260826-613e9a3199a5-c8809e1c8623'
PRIMARY_FAMILY = 'csn_quote_initiation_local_dislocation'
OPERATOR_ID = 'factor_mining_cross_sectional_microstructure_neighborhood_v1'
FIXED_PARAMS = {'family': 'csn_quote_initiation_local_dislocation',
 'signal': 'csn_taq_buy_initiation_neighbor_gap'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/csn_quote_initiation_local_dislocation/csn_taq_buy_initiation_neighbor_gap/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_cross_sectional_microstructure_neighborhood_v1/contract.json'


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
