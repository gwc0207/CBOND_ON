"""Healthy research-only view of the state-gated microstructure catalogue.

The source kernel remains the sole implementation and registration point.  A
completed aggregate-v2 smoke found every member of its
``floor_premium_gated_passive_depth_refresh`` family empty, while the other
three families were nonempty.  This explicit fail-closed view preserves the
original nine ``CatalogEntry`` objects and source kernel, and excludes only
those three audited empty signals.  It changes no field requirement, formula,
parameter, calculation, cache key, or time contract.

This wrapper is deliberately not imported by ``defs.__init__`` and is not in
the aggregate catalogue, a configuration, FactorStore, DB, scheduler, model,
or live path.  A later research composition must substitute it for (not append
it to) the original source catalogue because its entries intentionally reuse
the original signal names.
"""

from __future__ import annotations

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.defs import (
    research_factor_mining_intraday_state_gated_microstructure_v1 as _source,
)


CATALOG_VERSION = "20260803_intraday_state_gated_microstructure_healthy_catalog_v1"
SOURCE_MODULE_NAME = _source.__name__
KERNEL_NAME = _source.KERNEL_NAME

# Keep this ordered whitelist in the exact source catalogue order.  It makes
# source additions, removals, reordering, family changes, or kernel changes
# fail closed instead of silently widening a future research build.
HEALTHY_FAMILY_SIGNALS = {
    "duration_gated_quote_revision_calendar": (
        "isgm_duration_quote_revision_concentration",
        "isgm_duration_quote_revision_late_tilt",
        "isgm_duration_bid_ask_revision_clock_gap",
    ),
    "credit_gated_spread_spell_topology": (
        "isgm_credit_wide_spread_longest_spell_share",
        "isgm_credit_wide_spread_entry_clock",
        "isgm_credit_wide_spread_recovery_share",
    ),
    "stockvol_gated_trade_quote_clock_decoupling": (
        "isgm_stockvol_trade_quote_clock_center_gap",
        "isgm_stockvol_trade_quote_profile_cosine",
        "isgm_stockvol_silent_trade_share",
    ),
}
HEALTHY_FAMILIES = tuple(HEALTHY_FAMILY_SIGNALS)
HEALTHY_SIGNALS = tuple(
    signal for signals in HEALTHY_FAMILY_SIGNALS.values() for signal in signals
)

EXCLUDED_FAMILY = "floor_premium_gated_passive_depth_refresh"
EXCLUDED_SIGNALS = (
    "isgm_floorpremium_passive_depth_refresh_share",
    "isgm_floorpremium_passive_depth_side_balance",
    "isgm_floorpremium_passive_depth_late_tilt",
)

_SOURCE_SIGNALS = (
    *HEALTHY_FAMILY_SIGNALS["duration_gated_quote_revision_calendar"],
    *HEALTHY_FAMILY_SIGNALS["credit_gated_spread_spell_topology"],
    *EXCLUDED_SIGNALS,
    *HEALTHY_FAMILY_SIGNALS["stockvol_gated_trade_quote_clock_decoupling"],
)
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
            "healthy state-gated catalogue expected exactly "
            f"{len(_SOURCE_SIGNALS)} source entries, got {len(entries)}"
        )
    signals = tuple(entry.signal for entry in entries)
    if len(set(signals)) != len(signals):
        raise ValueError(
            "healthy state-gated catalogue source has duplicate signal names"
        )
    if signals != _SOURCE_SIGNALS:
        raise ValueError(
            "healthy state-gated catalogue source signal contract changed: "
            f"expected {_SOURCE_SIGNALS}, got {signals}"
        )
    if any(entry.kernel != KERNEL_NAME for entry in entries):
        raise ValueError(
            "healthy state-gated catalogue source has an unexpected kernel"
        )

    actual_family_by_signal = {entry.signal: entry.family for entry in entries}
    if actual_family_by_signal != _EXPECTED_FAMILY_BY_SIGNAL:
        raise ValueError("healthy state-gated catalogue source family contract changed")
    if (
        FactorRegistry.get(KERNEL_NAME)
        is not _source.FactorMiningIntradayStateGatedMicrostructureV1
    ):
        raise RuntimeError(
            "healthy state-gated catalogue source kernel is not correctly registered"
        )
    return entries


def intraday_state_gated_microstructure_healthy_catalog() -> tuple[
    _source.CatalogEntry, ...
]:
    """Return exactly the three audited nonempty source families."""

    entries = _validated_source_entries(tuple(_source.factor_mining_catalog()))
    healthy_entries = tuple(
        entry for entry in entries if entry.signal in _HEALTHY_SIGNAL_SET
    )
    if tuple(entry.signal for entry in healthy_entries) != HEALTHY_SIGNALS:
        raise RuntimeError(
            "healthy state-gated catalogue selection did not preserve source order"
        )
    if len({entry.signal for entry in healthy_entries}) != len(healthy_entries):
        raise RuntimeError(
            "healthy state-gated catalogue selection has duplicate signal names"
        )
    return healthy_entries


def factor_mining_catalog() -> tuple[_source.CatalogEntry, ...]:
    """Generic factor-mining runner entrypoint for the healthy subset."""

    return intraday_state_gated_microstructure_healthy_catalog()


__all__ = [
    "CATALOG_VERSION",
    "EXCLUDED_FAMILY",
    "EXCLUDED_SIGNALS",
    "HEALTHY_FAMILIES",
    "HEALTHY_FAMILY_SIGNALS",
    "HEALTHY_SIGNALS",
    "KERNEL_NAME",
    "SOURCE_MODULE_NAME",
    "factor_mining_catalog",
    "intraday_state_gated_microstructure_healthy_catalog",
]
