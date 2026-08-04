from __future__ import annotations

from collections import Counter

from cbond_on.domain.factors.defs import research_factor_mining_intraday_combined_v1 as combined


def test_combined_intraday_catalogue_preserves_all_families_and_unique_signals() -> None:
    entries = combined.factor_mining_catalog()

    assert len(entries) == 84
    assert len({entry.signal for entry in entries}) == 84
    assert len({entry.family for entry in entries}) == 13
    counts = Counter(entry.family for entry in entries)
    assert counts["open_gap_absorption"] == 6
    assert counts["intraday_phase_transition"] == 8
    assert counts["alpha_path_transform_port"] == 8
