"""Numerically audited research-only factor-mining aggregate catalogue.

v5 preserves the independently smoked v4 prefix and adds two later candidate
families only after their own strict-PIT/numerical-quality review. It remains a
research-only catalogue: no ``defs.__init__`` import, no live/model config,
and no production output, database, or scheduler integration.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.defs import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_aggregate_catalog_v4 as aggregate_v4,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1 as daily_credit_duration_term_supply_healthy,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_intraday_information_clock_geometry_v1 as intraday_information_clock_geometry,
)


CATALOG_VERSION = (
    "20260803_research_factor_mining_aggregate_catalog_v5_numerically_audited"
)
_REQUIRED_V4_VERSION = (
    "20260803_research_factor_mining_aggregate_catalog_v4_crossperiod_smoked"
)

# Preserve the audited v4 prefix exactly. The daily healthy view substitutes
# for its 12-signal v2 source (which is deliberately not in v4), rather than
# adding a duplicate source with the same signal names.
SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    *aggregate_v4.SOURCE_MODULES,
    ("intraday_information_clock_geometry", intraday_information_clock_geometry),
    (
        "daily_credit_duration_term_supply_healthy",
        daily_credit_duration_term_supply_healthy,
    ),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def _assert_v4_contract() -> None:
    if aggregate_v4.CATALOG_VERSION != _REQUIRED_V4_VERSION:
        raise RuntimeError(
            "aggregate v5 requires the immutable v4 core version "
            f"{_REQUIRED_V4_VERSION!r}, got {aggregate_v4.CATALOG_VERSION!r}"
        )


def factor_mining_aggregate_catalog() -> tuple[object, ...]:
    """Return the explicit numerically audited research catalogue."""

    _assert_v4_contract()
    return aggregate_v3._validate_source_catalogs(
        (source_name, module.factor_mining_catalog())
        for source_name, module in SOURCE_MODULES
    )


def factor_mining_catalog() -> tuple[object, ...]:
    """Generic factor-mining runner entrypoint."""

    return factor_mining_aggregate_catalog()
