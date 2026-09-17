"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'cabet_stock_reprice_bond_spread_recovery'
FACTOR_VERSION = 'catalog-20260826-715aad78c9f9-eb9636da2a58'
PRIMARY_FAMILY = 'cross_asset_triggered_bond_book_recovery'
OPERATOR_ID = 'factor_mining_cross_asset_book_event_transmission_v1'
FIXED_PARAMS = {'family': 'cross_asset_triggered_bond_book_recovery',
 'signal': 'cabet_stock_reprice_bond_spread_recovery'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/cross_asset_triggered_bond_book_recovery/cabet_stock_reprice_bond_spread_recovery/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_cross_asset_book_event_transmission_v1/contract.json'


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
