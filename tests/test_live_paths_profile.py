from __future__ import annotations

import os
from pathlib import Path

import pytest

from cbond_on.infra.live.config import configure_live_paths_profile


def test_configure_live_paths_profile_sets_explicit_profile(monkeypatch, tmp_path: Path) -> None:
    profile = tmp_path / "paths_live50_config.json5"
    profile.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("CBOND_ON_PATHS_CONFIG", raising=False)

    resolved = configure_live_paths_profile({"runtime": {"paths_config": str(profile)}})

    assert resolved == profile.resolve()
    assert Path(os.environ["CBOND_ON_PATHS_CONFIG"]).resolve() == profile.resolve()


def test_configure_live_paths_profile_rejects_conflicting_inherited_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    inherited = tmp_path / "paths_inherited_config.json5"
    requested = tmp_path / "paths_live50_config.json5"
    inherited.write_text("{}", encoding="utf-8")
    requested.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CBOND_ON_PATHS_CONFIG", str(inherited))

    with pytest.raises(RuntimeError, match="live paths profile conflict"):
        configure_live_paths_profile({"runtime": {"paths_config": str(requested)}})


def test_configure_live_paths_profile_is_noop_without_runtime(monkeypatch) -> None:
    monkeypatch.delenv("CBOND_ON_PATHS_CONFIG", raising=False)

    assert configure_live_paths_profile({}) is None
    assert os.environ.get("CBOND_ON_PATHS_CONFIG") is None
