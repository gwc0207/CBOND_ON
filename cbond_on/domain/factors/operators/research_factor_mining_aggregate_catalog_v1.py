"""Research-only aggregate catalogue for one-root factor-mining builds.

This module explicitly imports a fixed, already-tested set of sixteen research
factor modules.  Importing those modules executes their ``FactorRegistry``
decorators, so a future generic factor-mining build can load all kernels from a
single catalogue-module import.  It deliberately does not import from, or add
anything to, ``defs.__init__`` and is not used by any live configuration.

``factor_mining_catalog()`` concatenates source catalogues in the declared
order and returns their original entry objects unchanged.  It validates that a
signal appears exactly once and that a family belongs to exactly one source
module (while allowing the expected multiple signals within that family).
"""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType

from cbond_on.domain.factors.operators import (
    research_factor_mining_conditional_response_residual_v1 as conditional_response_residual,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_asset_book_geometry_v1 as cross_asset_book_geometry,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_debt_floor_wedge_dynamics_v1 as daily_debt_floor_wedge_dynamics,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_mark_barrier_dynamics_v1 as daily_mark_barrier_dynamics,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_interday_topology_v1 as daily_interday_topology,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_twap_microsegments_v1 as daily_twap_microsegments,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_execution_discreteness_v1 as intraday_execution_discreteness,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_state_gated_microstructure_v1 as intraday_state_gated_microstructure,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_conversion_parity_dynamics_v1 as intraday_conversion_parity_dynamics,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_lunch_reopen_dynamics_v1 as intraday_lunch_reopen_dynamics,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_temporal_shape_v1 as intraday_temporal_shape,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_transmission_response_v1 as intraday_transmission_response,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_trigger_state_response_v1 as intraday_trigger_state_response,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_limit_pressure_temporal_topology_v1 as limit_pressure_temporal_topology,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_quote_geometry_microprice_v1 as quote_geometry_microprice,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_underlying_cohort_intraday_rank_state_v1 as underlying_cohort_intraday_rank_state,
)


CATALOG_VERSION = "20260803_research_factor_mining_aggregate_catalog_v1"

# This order is intentional: it is the audited future one-root build order and
# must not silently expand to include any pending research module.
SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    ("conditional_response_residual", conditional_response_residual),
    ("intraday_temporal_shape", intraday_temporal_shape),
    ("intraday_transmission_response", intraday_transmission_response),
    ("quote_geometry_microprice", quote_geometry_microprice),
    ("intraday_lunch_reopen_dynamics", intraday_lunch_reopen_dynamics),
    ("cross_asset_book_geometry", cross_asset_book_geometry),
    ("daily_twap_microsegments", daily_twap_microsegments),
    ("daily_debt_floor_wedge_dynamics", daily_debt_floor_wedge_dynamics),
    ("daily_mark_barrier_dynamics", daily_mark_barrier_dynamics),
    ("daily_interday_topology", daily_interday_topology),
    ("intraday_execution_discreteness", intraday_execution_discreteness),
    ("intraday_state_gated_microstructure", intraday_state_gated_microstructure),
    ("limit_pressure_temporal_topology", limit_pressure_temporal_topology),
    ("intraday_conversion_parity_dynamics", intraday_conversion_parity_dynamics),
    ("underlying_cohort_intraday_rank_state", underlying_cohort_intraday_rank_state),
    ("intraday_trigger_state_response", intraday_trigger_state_response),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def _entry_field(entry: object, field: str, *, source: str) -> str:
    value = getattr(entry, field, None)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{source} catalogue entry has empty {field}: {entry!r}")
    return text


def _validate_source_catalogs(
    source_catalogs: Iterable[tuple[str, Iterable[object]]],
) -> tuple[object, ...]:
    """Return original entries after rejecting cross-source ambiguities."""

    seen_sources: set[str] = set()
    signal_owner: dict[str, tuple[str, str]] = {}
    family_owner: dict[str, tuple[str, str]] = {}
    entries: list[object] = []
    for source, raw_entries in source_catalogs:
        source_name = str(source).strip()
        if not source_name:
            raise ValueError("aggregate catalogue has an empty source name")
        if source_name in seen_sources:
            raise ValueError(f"aggregate catalogue has duplicate source module: {source_name}")
        seen_sources.add(source_name)
        source_entries = tuple(raw_entries)
        if not source_entries:
            raise ValueError(f"{source_name} factor_mining_catalog() returned no entries")
        for entry in source_entries:
            family = _entry_field(entry, "family", source=source_name)
            signal = _entry_field(entry, "signal", source=source_name)
            _entry_field(entry, "kernel", source=source_name)
            _entry_field(entry, "hypothesis", source=source_name)

            prior_signal = signal_owner.get(signal)
            if prior_signal is not None:
                prior_source, prior_family = prior_signal
                raise ValueError(
                    "aggregate catalogue has duplicate signal: "
                    f"{signal} in {prior_source}/{prior_family} and {source_name}/{family}"
                )
            signal_owner[signal] = (source_name, family)

            canonical_family = family.casefold()
            prior_family = family_owner.get(canonical_family)
            if prior_family is not None:
                prior_source, prior_name = prior_family
                if prior_name != family:
                    raise ValueError(
                        "aggregate catalogue has ambiguous family spelling: "
                        f"{prior_name!r} versus {family!r}"
                    )
                if prior_source != source_name:
                    raise ValueError(
                        "aggregate catalogue has duplicate family across sources: "
                        f"{family} in {prior_source} and {source_name}"
                    )
            else:
                family_owner[canonical_family] = (source_name, family)
            entries.append(entry)
    if not entries:
        raise ValueError("aggregate factor-mining catalogue is empty")
    return tuple(entries)


def factor_mining_aggregate_catalog() -> tuple[object, ...]:
    """Concatenate the fixed sixteen source catalogues in audited source order."""

    return _validate_source_catalogs(
        (source_name, module.factor_mining_catalog()) for source_name, module in SOURCE_MODULES
    )


def factor_mining_catalog() -> tuple[object, ...]:
    """Generic factor-mining runner entrypoint for the fixed aggregate."""

    return factor_mining_aggregate_catalog()
