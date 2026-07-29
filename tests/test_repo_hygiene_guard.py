from __future__ import annotations

from cbond_on.common.repo_hygiene_guard import _violations


def test_repo_hygiene_allows_config_runtime_module() -> None:
    assert _violations(["cbond_on/config/factor/runtime/default.json5"]) == []


def test_repo_hygiene_blocks_root_runtime_artifacts() -> None:
    assert _violations(["runtime/session_state.json"]) == ["runtime/session_state.json"]

