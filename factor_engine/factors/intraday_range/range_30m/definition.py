"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'range_30m'
FACTOR_VERSION = 'catalog-20260826-a7905bcb2b23-9ee4e7fefde7'
PRIMARY_FAMILY = 'intraday_range'
OPERATOR_ID = 'range_ratio'
FIXED_PARAMS = {'price_col': 'last', 'window_minutes': 30}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/range_30m'
CONTRACT_PATH = 'factors/intraday_range/range_30m/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/range_ratio/contract.json'


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
