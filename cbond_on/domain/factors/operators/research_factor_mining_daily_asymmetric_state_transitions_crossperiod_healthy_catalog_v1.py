"""Cross-period healthy research-only view of asymmetric daily states.

The existing seven-signal healthy wrapper is the sole audited upstream for
this second-level view.  Its strict-T1430 cross-period smoke found
``svcr_stress_gap_reversal_rate60`` entirely empty on 2025-04-02 and
2026-04-02, despite completed history being available.  The remaining yield
and duration/convexity families were nonempty and cross-sectionally
nonconstant on all three audited dates.

This module consequently exposes only those six original ``CatalogEntry``
objects.  It changes no kernel, formula, parameter, source field, cache key,
strict-T-1 alignment, or calculation.  Every upstream order, signal, family,
kernel, and registry identity is checked explicitly so source drift fails
closed instead of silently admitting an unaudited entry.

It is deliberately not imported by ``defs.__init__`` or any aggregate,
configuration, FactorStore, model, live, DB, or scheduler path.  A later
research composition must substitute this view for its audited upstream, not
append it, because its retained entries intentionally reuse signal names.
"""

from __future__ import annotations

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_asymmetric_state_transitions_healthy_catalog_v1 as _upstream,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_asymmetric_state_transitions_v1 as _source,
)


CATALOG_VERSION = (
    "20260803_daily_asymmetric_state_transitions_crossperiod_healthy_catalog_v1"
)
UPSTREAM_MODULE_NAME = _upstream.__name__
SOURCE_MODULE_NAME = _upstream.SOURCE_MODULE_NAME
KERNEL_NAME = _upstream.KERNEL_NAME

# This is an explicit family-first whitelist.  Never express the cross-period
# decision as an "all except" rule: a new upstream signal must not enter a
# future research build until it has its own cross-period audit.
CROSSPERIOD_HEALTHY_FAMILY_SIGNALS = {
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
}
CROSSPERIOD_HEALTHY_FAMILIES = tuple(CROSSPERIOD_HEALTHY_FAMILY_SIGNALS)
CROSSPERIOD_HEALTHY_SIGNALS = tuple(
    signal
    for signals in CROSSPERIOD_HEALTHY_FAMILY_SIGNALS.values()
    for signal in signals
)
EXCLUDED_UPSTREAM_SIGNALS = ("svcr_stress_gap_reversal_rate60",)

_UPSTREAM_FAMILY_SIGNALS = {
    **CROSSPERIOD_HEALTHY_FAMILY_SIGNALS,
    "prior_stockvol_candle_reversal_state": EXCLUDED_UPSTREAM_SIGNALS,
}
_UPSTREAM_FAMILIES = tuple(_UPSTREAM_FAMILY_SIGNALS)
_UPSTREAM_SIGNALS = tuple(
    signal for signals in _UPSTREAM_FAMILY_SIGNALS.values() for signal in signals
)
_CROSSPERIOD_SIGNAL_SET = frozenset(CROSSPERIOD_HEALTHY_SIGNALS)
_EXPECTED_UPSTREAM_FAMILY_BY_SIGNAL = {
    signal: family
    for family, signals in _UPSTREAM_FAMILY_SIGNALS.items()
    for signal in signals
}


def _validated_upstream_entries(
    entries: tuple[_source.CatalogEntry, ...],
) -> tuple[_source.CatalogEntry, ...]:
    """Reject every upstream-wrapper or source-kernel drift before selection."""

    if tuple(_upstream.HEALTHY_FAMILIES) != _UPSTREAM_FAMILIES:
        raise ValueError(
            "cross-period asymmetric-state upstream family contract changed: "
            f"expected {_UPSTREAM_FAMILIES}, got {_upstream.HEALTHY_FAMILIES}"
        )
    if tuple(_upstream.HEALTHY_SIGNALS) != _UPSTREAM_SIGNALS:
        raise ValueError(
            "cross-period asymmetric-state upstream signal contract changed: "
            f"expected {_UPSTREAM_SIGNALS}, got {_upstream.HEALTHY_SIGNALS}"
        )
    if dict(_upstream.HEALTHY_FAMILY_SIGNALS) != _UPSTREAM_FAMILY_SIGNALS:
        raise ValueError(
            "cross-period asymmetric-state upstream family-signal contract changed"
        )
    if _upstream.SOURCE_MODULE_NAME != _source.__name__:
        raise ValueError(
            "cross-period asymmetric-state upstream source module contract changed"
        )
    if _upstream.KERNEL_NAME != KERNEL_NAME:
        raise ValueError(
            "cross-period asymmetric-state upstream kernel contract changed"
        )
    if len(entries) != len(_UPSTREAM_SIGNALS):
        raise ValueError(
            "cross-period asymmetric-state upstream expected exactly "
            f"{len(_UPSTREAM_SIGNALS)} entries, got {len(entries)}"
        )
    signals = tuple(entry.signal for entry in entries)
    if len(set(signals)) != len(signals):
        raise ValueError(
            "cross-period asymmetric-state upstream has duplicate signal names"
        )
    if signals != _UPSTREAM_SIGNALS:
        raise ValueError(
            "cross-period asymmetric-state upstream entry order or signal contract changed: "
            f"expected {_UPSTREAM_SIGNALS}, got {signals}"
        )
    if any(entry.kernel != KERNEL_NAME for entry in entries):
        raise ValueError(
            "cross-period asymmetric-state upstream has an unexpected kernel"
        )
    if {
        entry.signal: entry.family for entry in entries
    } != _EXPECTED_UPSTREAM_FAMILY_BY_SIGNAL:
        raise ValueError(
            "cross-period asymmetric-state upstream family contract changed"
        )
    if (
        FactorRegistry.get(KERNEL_NAME)
        is not _source.FactorMiningDailyAsymmetricStateTransitionsV1
    ):
        raise RuntimeError(
            "cross-period asymmetric-state source kernel is not correctly registered"
        )
    return entries


def daily_asymmetric_state_transitions_crossperiod_healthy_catalog() -> tuple[
    _source.CatalogEntry, ...
]:
    """Return exactly the six cross-period healthy source entries."""

    entries = _validated_upstream_entries(tuple(_upstream.factor_mining_catalog()))
    selected = tuple(
        entry for entry in entries if entry.signal in _CROSSPERIOD_SIGNAL_SET
    )
    if tuple(entry.signal for entry in selected) != CROSSPERIOD_HEALTHY_SIGNALS:
        raise RuntimeError(
            "cross-period asymmetric-state selection did not preserve upstream order"
        )
    if len({entry.signal for entry in selected}) != len(selected):
        raise RuntimeError(
            "cross-period asymmetric-state selection has duplicate signal names"
        )
    return selected


def factor_mining_catalog() -> tuple[_source.CatalogEntry, ...]:
    """Generic factor-mining runner entrypoint for the audited subset."""

    return daily_asymmetric_state_transitions_crossperiod_healthy_catalog()


__all__ = [
    "CATALOG_VERSION",
    "CROSSPERIOD_HEALTHY_FAMILIES",
    "CROSSPERIOD_HEALTHY_FAMILY_SIGNALS",
    "CROSSPERIOD_HEALTHY_SIGNALS",
    "EXCLUDED_UPSTREAM_SIGNALS",
    "KERNEL_NAME",
    "SOURCE_MODULE_NAME",
    "UPSTREAM_MODULE_NAME",
    "daily_asymmetric_state_transitions_crossperiod_healthy_catalog",
    "factor_mining_catalog",
]
