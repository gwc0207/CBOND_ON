"""Healthy research-only view of asymmetric daily state transitions.

The source module remains the sole implementation and registration point.  In
the completed strict-T1430 two-score-day smoke,
``svcr_relief_gap_fade_rate60`` and
``svcr_stress_relief_reversal_spread60`` were empty on both days, whereas the
other seven source entries were nonempty and cross-sectionally nonconstant.
This explicit fail-closed view returns the original seven ``CatalogEntry``
objects and original kernel unchanged.  It changes no source field, strict-T-1
alignment, formula, parameter, cache key, or calculation.

It is deliberately not imported by ``defs.__init__`` or any aggregate,
configuration, FactorStore, model, live, DB, or scheduler path.  A future
research composition must substitute this view for the original source
catalogue, not append it, because the retained entries intentionally reuse
their source signal names.
"""

from __future__ import annotations

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_asymmetric_state_transitions_v1 as _source,
)


CATALOG_VERSION = "20260803_daily_asymmetric_state_transitions_healthy_catalog_v1"
SOURCE_MODULE_NAME = _source.__name__
KERNEL_NAME = _source.KERNEL_NAME

# This exact whitelist preserves the source's family-first order.  It is not
# an "all except" selector, so a later source addition cannot silently enter a
# research build.
HEALTHY_FAMILY_SIGNALS = {
    "prior_yield_directional_price_pass_through": (
        "ydpt_yield_rise_return_beta60",
        "ydpt_yield_fall_return_beta60",
        "ydpt_yield_return_beta_asymmetry60",
    ),
    "prior_duration_convexity_range_transition": (
        "dcrt_density_body_asymmetry60",
        "dcrt_density_range_asymmetry60",
        "dcrt_density_sign_agreement60",
    ),
    "prior_stockvol_candle_reversal_state": ("svcr_stress_gap_reversal_rate60",),
}
HEALTHY_FAMILIES = tuple(HEALTHY_FAMILY_SIGNALS)
HEALTHY_SIGNALS = tuple(
    signal for signals in HEALTHY_FAMILY_SIGNALS.values() for signal in signals
)
EXCLUDED_SIGNALS = (
    "svcr_relief_gap_fade_rate60",
    "svcr_stress_relief_reversal_spread60",
)

_SOURCE_SIGNALS = (
    *HEALTHY_FAMILY_SIGNALS["prior_yield_directional_price_pass_through"],
    *HEALTHY_FAMILY_SIGNALS["prior_duration_convexity_range_transition"],
    *HEALTHY_FAMILY_SIGNALS["prior_stockvol_candle_reversal_state"],
    *EXCLUDED_SIGNALS,
)
_HEALTHY_SIGNAL_SET = frozenset(HEALTHY_SIGNALS)
_EXPECTED_FAMILY_BY_SIGNAL = {
    signal: family
    for family, signals in HEALTHY_FAMILY_SIGNALS.items()
    for signal in signals
}
_EXPECTED_FAMILY_BY_SIGNAL.update(
    {signal: "prior_stockvol_candle_reversal_state" for signal in EXCLUDED_SIGNALS}
)


def _validated_source_entries(
    entries: tuple[_source.CatalogEntry, ...],
) -> tuple[_source.CatalogEntry, ...]:
    """Reject source-catalogue drift instead of silently changing this view."""

    if len(entries) != len(_SOURCE_SIGNALS):
        raise ValueError(
            "healthy asymmetric-state catalogue expected exactly "
            f"{len(_SOURCE_SIGNALS)} source entries, got {len(entries)}"
        )
    signals = tuple(entry.signal for entry in entries)
    if len(set(signals)) != len(signals):
        raise ValueError(
            "healthy asymmetric-state catalogue source has duplicate signal names"
        )
    if signals != _SOURCE_SIGNALS:
        raise ValueError(
            "healthy asymmetric-state catalogue source signal contract changed: "
            f"expected {_SOURCE_SIGNALS}, got {signals}"
        )
    if any(entry.kernel != KERNEL_NAME for entry in entries):
        raise ValueError(
            "healthy asymmetric-state catalogue source has an unexpected kernel"
        )
    if {entry.signal: entry.family for entry in entries} != _EXPECTED_FAMILY_BY_SIGNAL:
        raise ValueError(
            "healthy asymmetric-state catalogue source family contract changed"
        )
    if (
        FactorRegistry.get(KERNEL_NAME)
        is not _source.FactorMiningDailyAsymmetricStateTransitionsV1
    ):
        raise RuntimeError(
            "healthy asymmetric-state source kernel is not correctly registered"
        )
    return entries


def daily_asymmetric_state_transitions_healthy_catalog() -> tuple[
    _source.CatalogEntry, ...
]:
    """Return the seven audited nonempty, nonconstant source entries."""

    entries = _validated_source_entries(tuple(_source.factor_mining_catalog()))
    healthy_entries = tuple(
        entry for entry in entries if entry.signal in _HEALTHY_SIGNAL_SET
    )
    if tuple(entry.signal for entry in healthy_entries) != HEALTHY_SIGNALS:
        raise RuntimeError(
            "healthy asymmetric-state selection did not preserve source order"
        )
    if len({entry.signal for entry in healthy_entries}) != len(healthy_entries):
        raise RuntimeError(
            "healthy asymmetric-state selection has duplicate signal names"
        )
    return healthy_entries


def factor_mining_catalog() -> tuple[_source.CatalogEntry, ...]:
    """Generic factor-mining runner entrypoint for the healthy subset."""

    return daily_asymmetric_state_transitions_healthy_catalog()


__all__ = [
    "CATALOG_VERSION",
    "EXCLUDED_SIGNALS",
    "HEALTHY_FAMILIES",
    "HEALTHY_FAMILY_SIGNALS",
    "HEALTHY_SIGNALS",
    "KERNEL_NAME",
    "SOURCE_MODULE_NAME",
    "daily_asymmetric_state_transitions_healthy_catalog",
    "factor_mining_catalog",
]
