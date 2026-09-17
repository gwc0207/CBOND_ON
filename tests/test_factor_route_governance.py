from __future__ import annotations

from pathlib import Path

import pytest

import cbond_on.common.factor_route_governance_guard as guard
from cbond_on.core.config import load_paths_profile


def test_full_repository_factor_route_guard_has_no_unapproved_paths() -> None:
    assert guard.collect_violations() == []


def test_external_normal_paths_profile_cannot_bypass_canonical_route_validation(tmp_path: Path) -> None:
    profile = tmp_path / "untrusted.json5"
    profile.write_text(
        '{ raw_data_root: "x", clean_data_root: "y", results_root: "z" }',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical factor_table|audit_only"):
        load_paths_profile(profile)
