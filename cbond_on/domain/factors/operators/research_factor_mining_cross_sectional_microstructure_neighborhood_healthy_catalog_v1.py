"""Healthy research-only view of cross-sectional neighbourhood v1.

The original source kernel and its 12-signal catalogue remain immutable audit
evidence.  Its two-day strict-T1430 smoke found every member of
``csn_limit_queue_local_dislocation`` all-NaN, while the other three families
were nonempty and cross-sectionally nonconstant.  This module is therefore an
explicit fail-closed view: it imports the source kernel for registration,
returns the exact original ``CatalogEntry`` objects, and exposes only the nine
audited healthy signals.

It is deliberately excluded from ``defs.__init__``, aggregate catalogues,
factor contracts, configurations, FactorStores, DB writes, models, live paths,
and scheduler paths.  A later research composition must substitute this view
for, not append it to, the original source catalogue because signals retain
their original names.
"""

from __future__ import annotations

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_sectional_microstructure_neighborhood_v1 as _source,
)


CATALOG_VERSION = "20260803_cross_sectional_microstructure_neighborhood_healthy_catalog_v1"
SOURCE_MODULE_NAME = _source.__name__
KERNEL_NAME = _source.KERNEL_NAME

# Keep the whitelist in exact source order.  Any source addition, removal,
# reorder, family reassignment, or kernel change fails closed instead of
# silently changing a later research build.
EXCLUDED_FAMILY = "csn_limit_queue_local_dislocation"
EXCLUDED_SIGNALS = (
    "csn_lql_terminal_lock_neighbor_gap",
    "csn_lql_occupancy_neighbor_gap",
    "csn_lql_transition_neighbor_gap",
)
HEALTHY_FAMILY_SIGNALS = {
    "csn_passive_queue_local_dislocation": (
        "csn_pql_retention_neighbor_gap",
        "csn_pql_refill_neighbor_gap",
        "csn_pql_churn_neighbor_gap",
    ),
    "csn_quote_initiation_local_dislocation": (
        "csn_taq_buy_initiation_neighbor_gap",
        "csn_taq_sell_initiation_neighbor_gap",
        "csn_taq_imbalance_neighbor_gap",
    ),
    "csn_depth_cascade_local_dislocation": (
        "csn_dlc_bid_coherence_neighbor_gap",
        "csn_dlc_ask_coherence_neighbor_gap",
        "csn_dlc_crosslag_neighbor_gap",
    ),
}
HEALTHY_FAMILIES = tuple(HEALTHY_FAMILY_SIGNALS)
HEALTHY_SIGNALS = tuple(
    signal for signals in HEALTHY_FAMILY_SIGNALS.values() for signal in signals
)

_SOURCE_SIGNALS = (*EXCLUDED_SIGNALS, *HEALTHY_SIGNALS)
_HEALTHY_SIGNAL_SET = frozenset(HEALTHY_SIGNALS)
_EXPECTED_FAMILY_BY_SIGNAL = {
    signal: family
    for family, signals in HEALTHY_FAMILY_SIGNALS.items()
    for signal in signals
} | {signal: EXCLUDED_FAMILY for signal in EXCLUDED_SIGNALS}


def _validated_source_entries(
    entries: tuple[_source.CatalogEntry, ...],
) -> tuple[_source.CatalogEntry, ...]:
    """Reject source-catalogue drift instead of silently changing this view."""

    if len(entries) != len(_SOURCE_SIGNALS):
        raise ValueError(
            "healthy cross-sectional neighbourhood catalogue expected exactly "
            f"{len(_SOURCE_SIGNALS)} source entries, got {len(entries)}"
        )
    signals = tuple(entry.signal for entry in entries)
    if len(set(signals)) != len(signals):
        raise ValueError(
            "healthy cross-sectional neighbourhood catalogue source has duplicate signal names"
        )
    if signals != _SOURCE_SIGNALS:
        raise ValueError(
            "healthy cross-sectional neighbourhood catalogue source signal contract changed: "
            f"expected {_SOURCE_SIGNALS}, got {signals}"
        )
    if any(entry.kernel != KERNEL_NAME for entry in entries):
        raise ValueError(
            "healthy cross-sectional neighbourhood catalogue source has an unexpected kernel"
        )

    actual_family_by_signal = {entry.signal: entry.family for entry in entries}
    if actual_family_by_signal != _EXPECTED_FAMILY_BY_SIGNAL:
        raise ValueError(
            "healthy cross-sectional neighbourhood catalogue source family contract changed"
        )
    if (
        FactorRegistry.get(KERNEL_NAME)
        is not _source.FactorMiningCrossSectionalMicrostructureNeighborhoodV1
    ):
        raise RuntimeError(
            "healthy cross-sectional neighbourhood catalogue source kernel is not correctly registered"
        )
    return entries


def cross_sectional_microstructure_neighborhood_healthy_catalog() -> tuple[
    _source.CatalogEntry, ...
]:
    """Return exactly the three audited nonempty source families."""

    entries = _validated_source_entries(tuple(_source.factor_mining_catalog()))
    healthy_entries = tuple(
        entry for entry in entries if entry.signal in _HEALTHY_SIGNAL_SET
    )
    if tuple(entry.signal for entry in healthy_entries) != HEALTHY_SIGNALS:
        raise RuntimeError(
            "healthy cross-sectional neighbourhood catalogue selection did not preserve source order"
        )
    if len({entry.signal for entry in healthy_entries}) != len(healthy_entries):
        raise RuntimeError(
            "healthy cross-sectional neighbourhood catalogue selection has duplicate signal names"
        )
    return healthy_entries


def factor_mining_catalog() -> tuple[_source.CatalogEntry, ...]:
    """Generic factor-mining runner entrypoint for the healthy subset."""

    return cross_sectional_microstructure_neighborhood_healthy_catalog()


__all__ = [
    "CATALOG_VERSION",
    "EXCLUDED_FAMILY",
    "EXCLUDED_SIGNALS",
    "HEALTHY_FAMILIES",
    "HEALTHY_FAMILY_SIGNALS",
    "HEALTHY_SIGNALS",
    "KERNEL_NAME",
    "SOURCE_MODULE_NAME",
    "cross_sectional_microstructure_neighborhood_healthy_catalog",
    "factor_mining_catalog",
]
