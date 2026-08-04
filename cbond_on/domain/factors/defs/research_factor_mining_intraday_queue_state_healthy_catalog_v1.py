"""Healthy research-only view of the intraday queue-state catalogue.

The source queue-state kernel contains three lock-state signals which were
shown to be cross-sectionally constant in the research build.  This module is
an explicit, fail-closed catalogue view: it imports the already-tested source
kernel for registration, preserves its original entry objects, and exposes
only the nine non-lock signals.  It is deliberately not imported by
``defs.__init__`` and has no live, configuration, I/O, or FactorStore effect.
"""

from __future__ import annotations

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.defs import research_factor_mining_intraday_queue_state_v1 as _queue


CATALOG_VERSION = "20260803_intraday_queue_state_healthy_catalog_v1"
SOURCE_KERNEL_NAME = _queue.KERNEL_NAME
EXCLUDED_FAMILY = "limit_queue_lock_state_machine"
EXCLUDED_SIGNALS = (
    "lqls_signed_terminal_lock",
    "lqls_lock_occupancy_share",
    "lqls_state_transition_rate",
)
HEALTHY_FAMILY_SIGNALS = {
    "passive_queue_lifecycle_after_trade": (
        "pql_trade_depth_retention",
        "pql_trade_refill_imbalance",
        "pql_trade_queue_churn_per_trade",
    ),
    "trade_at_quote_initiation_state": (
        "taqi_buy_quote_initiation_share",
        "taqi_sell_quote_initiation_share",
        "taqi_signed_quote_initiation_imbalance",
    ),
    "depth_layer_update_cascade": (
        "dluc_bid_adjacent_update_coherence",
        "dluc_ask_adjacent_update_coherence",
        "dluc_cross_side_update_lag_asymmetry",
    ),
}
HEALTHY_SIGNALS = tuple(
    signal
    for signals in HEALTHY_FAMILY_SIGNALS.values()
    for signal in signals
)
_SOURCE_SIGNALS = (*EXCLUDED_SIGNALS, *HEALTHY_SIGNALS)
_HEALTHY_SIGNAL_SET = frozenset(HEALTHY_SIGNALS)


def _validated_source_entries(entries: tuple[_queue.CatalogEntry, ...]) -> tuple[_queue.CatalogEntry, ...]:
    """Reject a source-catalogue drift instead of silently changing this view."""

    if len(entries) != len(_SOURCE_SIGNALS):
        raise ValueError(
            "healthy queue-state catalogue expected exactly "
            f"{len(_SOURCE_SIGNALS)} source entries, got {len(entries)}"
        )
    signals = tuple(entry.signal for entry in entries)
    if len(set(signals)) != len(signals):
        raise ValueError("healthy queue-state catalogue source has duplicate signal names")
    if signals != _SOURCE_SIGNALS:
        raise ValueError(
            "healthy queue-state catalogue source signal contract changed: "
            f"expected {_SOURCE_SIGNALS}, got {signals}"
        )
    if any(entry.kernel != SOURCE_KERNEL_NAME for entry in entries):
        raise ValueError("healthy queue-state catalogue source has an unexpected kernel")

    expected_family_by_signal = {
        signal: EXCLUDED_FAMILY for signal in EXCLUDED_SIGNALS
    } | {
        signal: family
        for family, signals_for_family in HEALTHY_FAMILY_SIGNALS.items()
        for signal in signals_for_family
    }
    actual_family_by_signal = {entry.signal: entry.family for entry in entries}
    if actual_family_by_signal != expected_family_by_signal:
        raise ValueError("healthy queue-state catalogue source family contract changed")
    if FactorRegistry.get(SOURCE_KERNEL_NAME) is not _queue.FactorMiningIntradayQueueStateV1:
        raise RuntimeError("healthy queue-state catalogue source kernel is not correctly registered")
    return entries


def factor_mining_intraday_queue_state_healthy_catalog_v1_catalog() -> tuple[_queue.CatalogEntry, ...]:
    """Return only the three validated non-lock queue-state families."""

    entries = _validated_source_entries(tuple(_queue.factor_mining_catalog()))
    healthy_entries = tuple(entry for entry in entries if entry.signal in _HEALTHY_SIGNAL_SET)
    if tuple(entry.signal for entry in healthy_entries) != HEALTHY_SIGNALS:
        raise RuntimeError("healthy queue-state catalogue selection did not preserve its exact signal order")
    if len({entry.signal for entry in healthy_entries}) != len(healthy_entries):
        raise RuntimeError("healthy queue-state catalogue selection has duplicate signal names")
    return healthy_entries


def factor_mining_catalog() -> tuple[_queue.CatalogEntry, ...]:
    """Generic factor-mining runner entrypoint for the healthy queue-state view."""

    return factor_mining_intraday_queue_state_healthy_catalog_v1_catalog()
