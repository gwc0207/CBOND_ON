"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_intraday_information_clock_geometry_v1'
OPERATOR_VERSION = '20260803_intraday_information_clock_geometry_v1'
OPERATOR_CLASS = 'FactorMiningIntradayInformationClockGeometryV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_intraday_information_clock_geometry_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_intraday_information_clock_geometry_v1.py'
IMPLEMENTATION_SHA256 = 'c3de662d1c32f0089b24785d0e3315852bed75866893a729eb82529dc49f1916'
CONTRACT_PATH = 'operators/factor_mining_intraday_information_clock_geometry_v1/contract.json'


def definition_payload() -> dict[str, object]:
    """Return immutable identity metadata without importing the runtime implementation."""

    return {
        'operator_id': OPERATOR_ID,
        'operator_version': OPERATOR_VERSION,
        'operator_class': OPERATOR_CLASS,
        'implementation_module': IMPLEMENTATION_MODULE,
        'implementation_path': IMPLEMENTATION_PATH,
        'implementation_sha256': IMPLEMENTATION_SHA256,
        'contract_path': CONTRACT_PATH,
    }
