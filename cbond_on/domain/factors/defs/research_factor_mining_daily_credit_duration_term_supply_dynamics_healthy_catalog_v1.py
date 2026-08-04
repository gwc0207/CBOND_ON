"""Healthy research-only view of v2 credit-duration-term-supply dynamics.

The immutable v2 source kernel produced a structurally valid two-score-day
strict-T1430 smoke.  Its three ``*_standardized_oos20`` residuals nevertheless
showed material current-observation scale excursions, including an absolute
value of 11179.8078 for ``tsrr2_supply_term_standardized_oos20``.  This
explicit, fail-closed catalogue view therefore retains only the other nine
source entries, while preserving all four source families.

This module only selects original ``CatalogEntry`` objects.  It does not alter
the source formulae, parameters, required fields, strict-T-1 alignment, cache
keys, computation, or registration.  It is deliberately not imported by
``defs.__init__`` or any aggregate, configuration, FactorStore, model, live,
DB, or scheduler path.  A later research composition must substitute this
view for the v2 source catalogue, rather than append it, because the retained
entries intentionally reuse their source signal names.
"""

from __future__ import annotations

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_credit_duration_term_supply_dynamics_v2 as _source,
)


CATALOG_VERSION = (
    "20260803_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1"
)
SOURCE_MODULE_NAME = _source.__name__
SOURCE_CATALOG_VERSION = _source.CATALOG_VERSION
_EXPECTED_SOURCE_CATALOG_VERSION = (
    "20260803_daily_credit_duration_term_supply_dynamics_v2"
)
KERNEL_NAME = _source.KERNEL_NAME

# The explicit whitelist preserves the source's family-first ordering.  It is
# intentionally not expressed as an "all except" selector, so a later v2
# source addition cannot silently enter a research build.
HEALTHY_FAMILY_SIGNALS = {
    "prior_credit_duration_convexity_motion_robust": (
        "cdmr_yield_duration_delta_corr20",
        "cdmr_yield_convexity_delta_corr20",
        "cdmr_duration_convexity_delta_corr20",
    ),
    "prior_term_supply_roll_rebalancing_robust": (
        "tsrr2_supply_term_level_corr20",
        "tsrr2_duration_term_ratio_delta5",
    ),
    "prior_credit_flow_absorption_dynamics_robust": (
        "cfar_yield_amount_capacity_delta_corr20",
        "cfar_yield_volume_capacity_delta_corr20",
    ),
    "prior_convexity_anchor_transition_robust": (
        "catr_convexity_duration_delta_corr20",
        "catr_convexity_anchor_delta_corr20",
    ),
}
HEALTHY_FAMILIES = tuple(HEALTHY_FAMILY_SIGNALS)
HEALTHY_SIGNALS = tuple(
    signal for signals in HEALTHY_FAMILY_SIGNALS.values() for signal in signals
)
EXCLUDED_SIGNALS = (
    "tsrr2_supply_term_standardized_oos20",
    "cfar_yield_price_supply_standardized_oos20",
    "catr_convexity_state_standardized_oos20",
)

_SOURCE_FAMILY_SIGNALS = {
    "prior_credit_duration_convexity_motion_robust": (
        "cdmr_yield_duration_delta_corr20",
        "cdmr_yield_convexity_delta_corr20",
        "cdmr_duration_convexity_delta_corr20",
    ),
    "prior_term_supply_roll_rebalancing_robust": (
        "tsrr2_supply_term_level_corr20",
        "tsrr2_duration_term_ratio_delta5",
        "tsrr2_supply_term_standardized_oos20",
    ),
    "prior_credit_flow_absorption_dynamics_robust": (
        "cfar_yield_amount_capacity_delta_corr20",
        "cfar_yield_volume_capacity_delta_corr20",
        "cfar_yield_price_supply_standardized_oos20",
    ),
    "prior_convexity_anchor_transition_robust": (
        "catr_convexity_duration_delta_corr20",
        "catr_convexity_anchor_delta_corr20",
        "catr_convexity_state_standardized_oos20",
    ),
}
_SOURCE_FAMILIES = tuple(_SOURCE_FAMILY_SIGNALS)
_SOURCE_SIGNALS = tuple(
    signal for signals in _SOURCE_FAMILY_SIGNALS.values() for signal in signals
)
_HEALTHY_SIGNAL_SET = frozenset(HEALTHY_SIGNALS)
_EXPECTED_SOURCE_FAMILY_BY_SIGNAL = {
    signal: family
    for family, signals in _SOURCE_FAMILY_SIGNALS.items()
    for signal in signals
}


def _validated_source_entries(
    entries: tuple[_source.CatalogEntry, ...],
) -> tuple[_source.CatalogEntry, ...]:
    """Reject every v2 source-catalogue drift before selecting entries."""

    if _source.CATALOG_VERSION != _EXPECTED_SOURCE_CATALOG_VERSION:
        raise ValueError(
            "healthy credit-duration-term-supply source catalogue version changed: "
            f"expected {_EXPECTED_SOURCE_CATALOG_VERSION}, got {_source.CATALOG_VERSION}"
        )
    if len(entries) != len(_SOURCE_SIGNALS):
        raise ValueError(
            "healthy credit-duration-term-supply catalogue expected exactly "
            f"{len(_SOURCE_SIGNALS)} source entries, got {len(entries)}"
        )
    signals = tuple(entry.signal for entry in entries)
    if len(set(signals)) != len(signals):
        raise ValueError(
            "healthy credit-duration-term-supply source has duplicate signal names"
        )
    if signals != _SOURCE_SIGNALS:
        raise ValueError(
            "healthy credit-duration-term-supply source signal contract changed: "
            f"expected {_SOURCE_SIGNALS}, got {signals}"
        )
    if any(entry.kernel != KERNEL_NAME for entry in entries):
        raise ValueError(
            "healthy credit-duration-term-supply source has an unexpected kernel"
        )
    if {
        entry.signal: entry.family for entry in entries
    } != _EXPECTED_SOURCE_FAMILY_BY_SIGNAL:
        raise ValueError(
            "healthy credit-duration-term-supply source family contract changed"
        )
    if (
        FactorRegistry.get(KERNEL_NAME)
        is not _source.FactorMiningDailyCreditDurationTermSupplyDynamicsV2
    ):
        raise RuntimeError(
            "healthy credit-duration-term-supply source kernel is not correctly registered"
        )
    return entries


def daily_credit_duration_term_supply_dynamics_healthy_catalog() -> tuple[
    _source.CatalogEntry, ...
]:
    """Return exactly the nine auditable, numerically healthy v2 entries."""

    entries = _validated_source_entries(tuple(_source.factor_mining_catalog()))
    healthy_entries = tuple(
        entry for entry in entries if entry.signal in _HEALTHY_SIGNAL_SET
    )
    if tuple(entry.signal for entry in healthy_entries) != HEALTHY_SIGNALS:
        raise RuntimeError(
            "healthy credit-duration-term-supply selection did not preserve source order"
        )
    if len({entry.signal for entry in healthy_entries}) != len(healthy_entries):
        raise RuntimeError(
            "healthy credit-duration-term-supply selection has duplicate signal names"
        )
    return healthy_entries


def factor_mining_catalog() -> tuple[_source.CatalogEntry, ...]:
    """Generic factor-mining runner entrypoint for the healthy v2 subset."""

    return daily_credit_duration_term_supply_dynamics_healthy_catalog()


__all__ = [
    "CATALOG_VERSION",
    "EXCLUDED_SIGNALS",
    "HEALTHY_FAMILIES",
    "HEALTHY_FAMILY_SIGNALS",
    "HEALTHY_SIGNALS",
    "KERNEL_NAME",
    "SOURCE_CATALOG_VERSION",
    "SOURCE_MODULE_NAME",
    "daily_credit_duration_term_supply_dynamics_healthy_catalog",
    "factor_mining_catalog",
]
