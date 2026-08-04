"""Research-only healthy wrapper for the audited cross-asset event-clock v1.

The underlying v1 kernel and its six-signal artefact remain immutable for
audit.  This catalogue exposes only the five signals that were nondegenerate
in the completed six-score-day smoke, deliberately excluding
"xca_lull_overlap_excess" because it was approximately 99.867% zero.

It has no factor implementation, configuration, I/O, or live effect.  The
returned entries are the exact original CatalogEntry objects, so the existing
strict-PIT kernel, formulae, and family assignments remain unchanged.  This
module is intentionally excluded from defs.__init__ and from the aggregate
catalogue until a separate owner-approved composition step.
"""

from __future__ import annotations

from cbond_on.domain.factors.defs import research_factor_mining_cross_asset_event_clock_v1 as _source


CATALOG_VERSION = "20260803_cross_asset_event_clock_healthy_v1"
SOURCE_MODULE = _source.__name__
SOURCE_KERNEL_NAME = _source.KERNEL_NAME

# This is an explicit audited whitelist, rather than a generic "all except"
# filter, so later source-catalogue additions cannot silently enter a build.
HEALTHY_SIGNALS = (
    "xca_event_burst_coactivity",
    "xca_stock_leads_bond_event_clock",
    "xca_signed_execution_agreement",
    "xca_stock_leads_bond_signed_execution",
    "xca_signed_execution_intensity_gap",
)
EXCLUDED_SIGNALS = frozenset({"xca_lull_overlap_excess"})
_AUDITED_SOURCE_SIGNAL_ORDER = (
    "xca_event_burst_coactivity",
    "xca_stock_leads_bond_event_clock",
    "xca_lull_overlap_excess",
    "xca_signed_execution_agreement",
    "xca_stock_leads_bond_signed_execution",
    "xca_signed_execution_intensity_gap",
)


def _validated_source_catalog() -> tuple[_source.CatalogEntry, ...]:
    """Return the fixed audited source catalogue or fail closed on drift."""

    entries = tuple(_source.factor_mining_catalog())
    signals = tuple(entry.signal for entry in entries)
    if signals != _AUDITED_SOURCE_SIGNAL_ORDER:
        raise ValueError(
            "cross-asset event-clock healthy catalogue source catalogue drift: "
            f"expected {_AUDITED_SOURCE_SIGNAL_ORDER}, got {signals}"
        )
    return entries


def factor_mining_cross_asset_event_clock_healthy_catalog() -> tuple[_source.CatalogEntry, ...]:
    """Return the audited nondegenerate source entries in original order."""

    return tuple(entry for entry in _validated_source_catalog() if entry.signal in HEALTHY_SIGNALS)


def factor_mining_catalog() -> tuple[_source.CatalogEntry, ...]:
    """Generic research-expansion entrypoint for the healthy wrapper."""

    return factor_mining_cross_asset_event_clock_healthy_catalog()
