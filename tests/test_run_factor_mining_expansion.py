from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

import harness.tools.run_factor_mining_expansion as expansion_runner


EXPANSION_MODULE = "cbond_on.domain.factors.defs.research_factor_mining_expansion_intraday_v1"


def _arguments(scratch_root: Path, *extra: str) -> list[str]:
    return [
        "--catalog-module",
        EXPANSION_MODULE,
        "--scratch-root",
        str(scratch_root),
        "--start",
        "2025-01-01",
        "--end",
        "2026-07-30",
        *extra,
    ]


def test_default_preflight_imports_research_registration_and_creates_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    """The real frozen profile is readable without creating the supplied root."""

    scratch_root = Path(r"D:/cbond_on/research_scratch") / f"_pytest_expansion_preflight_{uuid.uuid4().hex}"
    assert not scratch_root.exists()

    assert expansion_runner.main(_arguments(scratch_root)) == 0

    stdout = capsys.readouterr().out
    assert '"execute_requested": false' in stdout
    assert '"signals": 60' in stdout
    assert '"families": 10' in stdout
    assert '"engine": "python"' in stdout
    assert '"panel_source": "clean_direct"' in stdout
    assert not scratch_root.exists()


def test_scratch_root_must_be_strict_child_and_absent_for_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(expansion_runner, "_RESEARCH_SCRATCH_PARENT", scratch_parent)

    with pytest.raises(ValueError, match="strictly below"):
        expansion_runner._assert_scratch_root(scratch_parent, execute=False)

    scratch_root = scratch_parent / "candidate"
    assert expansion_runner._assert_scratch_root(scratch_root, execute=False) == scratch_root.resolve()
    scratch_root.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="previously absent"):
        expansion_runner._assert_scratch_root(scratch_root, execute=True)


def test_execute_writes_evidence_only_after_a_successful_mocked_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    scratch_root = scratch_parent / "candidate"
    monkeypatch.setattr(expansion_runner, "_RESEARCH_SCRATCH_PARENT", scratch_parent)

    def fake_run_factor_batch(cfg: dict, *, paths_cfg: dict) -> Path:
        assert cfg["backtest_enabled"] is False
        assert cfg["screening"]["enabled"] is False
        assert cfg["bad_factor_report"]["enabled"] is False
        assert paths_cfg["raw_data_root"] == "D:/cbond_data_hub/raw_data"
        assert paths_cfg["clean_data_root"] == "D:/cbond_data_hub/clean_data"
        assert Path(paths_cfg["factor_data_root"]).resolve().is_relative_to(scratch_root.resolve())
        out_root = scratch_root / "results" / "2025-01-01_2026-07-30" / "Single_Factor" / "mocked"
        out_root.mkdir(parents=True)
        return out_root

    monkeypatch.setattr(expansion_runner, "run_factor_batch", fake_run_factor_batch)

    assert expansion_runner.main(_arguments(scratch_root, "--execute")) == 0

    out_root = scratch_root / "results" / "2025-01-01_2026-07-30" / "Single_Factor" / "mocked"
    family_catalog = json.loads((out_root / "factor_mining_family_catalog.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_root / "factor_mining_run_manifest.json").read_text(encoding="utf-8"))
    assert len(family_catalog["families"]) == 10
    assert manifest["research_only"] is True
    assert manifest["catalogue"]["module"] == EXPANSION_MODULE
    assert manifest["derived_write_roots"]["factor_data_root"] == (scratch_root / "factor_data").resolve().as_posix()


def test_failed_mocked_batch_cannot_emit_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    scratch_root = scratch_parent / "candidate"
    monkeypatch.setattr(expansion_runner, "_RESEARCH_SCRATCH_PARENT", scratch_parent)

    def failed_run_factor_batch(cfg: dict, *, paths_cfg: dict) -> Path:
        raise RuntimeError("simulated build failure")

    monkeypatch.setattr(expansion_runner, "run_factor_batch", failed_run_factor_batch)

    with pytest.raises(RuntimeError, match="simulated build failure"):
        expansion_runner.main(_arguments(scratch_root, "--execute"))
    assert not scratch_root.exists()
