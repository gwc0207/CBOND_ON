"""Cross-period-smoked, research-only factor-mining aggregate catalogue.

v4 extends the smoke-healthy v3 core with independently verified sources.  It
uses only research registrations and is intentionally absent from
``defs.__init__``, live configurations, model configurations, database paths,
and scheduler paths.  A source is listed only after its original or healthy
wrapper has clear strict-PIT smoke evidence; the global IC and correlation
screen remains the final admission gate.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_asset_activity_calendar_v1 as cross_asset_activity_calendar,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_asset_book_event_transmission_v1 as cross_asset_book_event_transmission,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_asset_event_sequence_topology_v1 as cross_asset_event_sequence_topology,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_sectional_microstructure_neighborhood_healthy_catalog_v1 as cross_sectional_microstructure_neighborhood_healthy,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_asymmetric_state_transitions_crossperiod_healthy_catalog_v1 as daily_asymmetric_state_transitions_crossperiod_healthy,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_price_base_relations_v1 as daily_price_base_relations,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_historical_execution_to_open_transition_state_v1 as historical_execution_to_open_transition_state,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_queue_state_healthy_catalog_v1 as intraday_queue_state_healthy,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_trade_grid_topology_v1 as intraday_trade_grid_topology,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_quote_execution_dynamics_v1 as quote_execution_dynamics,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_structural_neighborhood_v1 as structural_neighborhood,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_underlying_cohort_distribution_v1 as underlying_cohort_distribution,
)


CATALOG_VERSION = (
    "20260803_research_factor_mining_aggregate_catalog_v4_crossperiod_smoked"
)
_REQUIRED_V3_VERSION = "20260803_research_factor_mining_aggregate_catalog_v3_healthy"

# Keep v3 intact as an audited prefix.  The source order is a data contract:
# each later addition has its own smoke evidence and must not be silently
# replaced by a source with the same economic name.
SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    *aggregate_v3.SOURCE_MODULES,
    ("daily_price_base_relations", daily_price_base_relations),
    ("cross_asset_book_event_transmission", cross_asset_book_event_transmission),
    (
        "historical_execution_to_open_transition_state",
        historical_execution_to_open_transition_state,
    ),
    (
        "daily_asymmetric_state_transitions_crossperiod_healthy",
        daily_asymmetric_state_transitions_crossperiod_healthy,
    ),
    ("intraday_trade_grid_topology", intraday_trade_grid_topology),
    ("cross_asset_activity_calendar", cross_asset_activity_calendar),
    ("cross_asset_event_sequence_topology", cross_asset_event_sequence_topology),
    ("quote_execution_dynamics", quote_execution_dynamics),
    (
        "cross_sectional_microstructure_neighborhood_healthy",
        cross_sectional_microstructure_neighborhood_healthy,
    ),
    ("structural_neighborhood", structural_neighborhood),
    ("underlying_cohort_distribution", underlying_cohort_distribution),
    ("intraday_queue_state_healthy", intraday_queue_state_healthy),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)

# These modules deliberately remain outside SOURCE_MODULES: their smoke
# evidence is not yet sufficient for a full-window build.
_EXCLUDED_UNTIL_FURTHER_EVIDENCE = (
    "cbond_on.domain.factors.operators."
    "research_factor_mining_cross_sectional_rank_lattice_healthy_catalog_v1",
)


def _assert_v3_contract() -> None:
    if aggregate_v3.CATALOG_VERSION != _REQUIRED_V3_VERSION:
        raise RuntimeError(
            "aggregate v4 requires the immutable v3 core version "
            f"{_REQUIRED_V3_VERSION!r}, got {aggregate_v3.CATALOG_VERSION!r}"
        )


def factor_mining_aggregate_catalog() -> tuple[object, ...]:
    """Return the explicit cross-period-smoked research catalogue."""

    _assert_v3_contract()
    return aggregate_v3._validate_source_catalogs(
        (source_name, module.factor_mining_catalog())
        for source_name, module in SOURCE_MODULES
    )


def factor_mining_catalog() -> tuple[object, ...]:
    """Generic expansion-runner entrypoint."""

    return factor_mining_aggregate_catalog()
