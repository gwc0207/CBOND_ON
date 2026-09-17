from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from cbond_on.app.usecases import factor_build_runtime
from cbond_on.domain.factors import operator_default_loader


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _fresh_probe(source: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-B", "-c", source],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payloads = [
        line.removeprefix("LEGACY_LOADER_PROBE=")
        for line in result.stdout.splitlines()
        if line.startswith("LEGACY_LOADER_PROBE=")
    ]
    assert len(payloads) == 1, result.stdout
    return json.loads(payloads[0])


def test_fresh_operator_package_is_empty_until_explicit_profile_loader() -> None:
    payload = _fresh_probe(
        """
import json

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors import operators
from cbond_on.domain.factors.operator_default_loader import (
    DEFAULT_OPERATOR_COUNT,
    load_default_operators,
)

before = len(list(FactorRegistry.names()))
before_has_aacb = hasattr(operators, "AacbFactor")
first_modules = load_default_operators()
after = len(list(FactorRegistry.names()))
second_modules = load_default_operators()
print("LEGACY_LOADER_PROBE=" + json.dumps({
    "before": before,
    "before_has_aacb": before_has_aacb,
    "declared_modules": DEFAULT_OPERATOR_COUNT,
    "first_modules": len(first_modules),
    "after": after,
    "second_modules": len(second_modules),
    "after_has_aacb": hasattr(operators, "AacbFactor"),
}))
"""
    )

    assert payload == {
        "after": 194,
        "after_has_aacb": True,
        "before": 0,
        "before_has_aacb": False,
        "declared_modules": 194,
        "first_modules": 194,
        "second_modules": 194,
    }


def test_fresh_live_factor_build_runtime_does_not_import_default_operator_loader() -> None:
    payload = _fresh_probe(
        """
import json
import sys

from cbond_on.app.usecases import factor_build_runtime  # noqa: F401
from cbond_on.core.registry import OperatorRegistry

print("LEGACY_LOADER_PROBE=" + json.dumps({
    "loader_imported": "cbond_on.domain.factors.operator_default_loader" in sys.modules,
    "registered": len(list(OperatorRegistry.names())),
}))
"""
    )

    assert payload == {"loader_imported": False, "registered": 0}


def test_factor_build_runtime_rejects_legacy_direct_factor_writer_without_live_admission(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_loader() -> tuple[str, ...]:
        calls.append("legacy_loader")
        return ()

    class _PipelineResult:
        written = 0
        skipped = 0

    def fake_config(name: str) -> dict[str, str]:
        if name == "paths":
            return {"panel_data_root": "panel", "factor_data_root": "factor"}
        if name == "panel":
            return {}
        raise AssertionError(f"unexpected config request: {name}")

    monkeypatch.setattr(factor_build_runtime, "load_config_file", fake_config)
    monkeypatch.setattr(factor_build_runtime, "validate_factor_execution_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(operator_default_loader, "load_default_operators", fake_loader)
    monkeypatch.setattr(factor_build_runtime, "build_signal_specs", lambda _cfg: [])
    monkeypatch.setattr(
        factor_build_runtime,
        "prepare_live50_factor_admission",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        factor_build_runtime,
        "issue_live50_factor_store_write_permit",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(factor_build_runtime, "run_factor_pipeline", lambda *_args, **_kwargs: _PipelineResult())

    base_cfg = {
        "start": "2026-08-20",
        "end": "2026-08-20",
        "panel_name": "T1430",
    }
    with pytest.raises(RuntimeError, match="canonical factor_table live-writer"):
        factor_build_runtime.run(cfg={**base_cfg, "live_factor_admission": {}})
    with pytest.raises(RuntimeError, match="canonical factor_table live-writer"):
        factor_build_runtime.run(cfg=base_cfg)
    assert calls == []
