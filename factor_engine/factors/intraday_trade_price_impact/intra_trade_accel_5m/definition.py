"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'intra_trade_accel_5m'
FACTOR_VERSION = 'catalog-20260826-a31605881ba6-2de7fd0e2f90'
PRIMARY_FAMILY = 'intraday_trade_price_impact'
OPERATOR_ID = 'factor_mining_intraday_catalog_v1'
FIXED_PARAMS = {'family': 'intraday_trade_price_impact', 'signal': 'intra_trade_accel_5m'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/intraday_trade_price_impact/intra_trade_accel_5m/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_intraday_catalog_v1/contract.json'


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
