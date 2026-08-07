from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

import harness.tools.run_factor_mining_expansion as expansion_runner
from cbond_on.domain.factors.spec import FactorSpec


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


def test_default_preflight_rejects_an_unported_python_catalogue_without_creating_root() -> None:
    """The old catalogue cannot be relaunched as a Python factor experiment."""

    scratch_root = Path(r"D:/cbond_on/research_scratch") / f"_pytest_expansion_preflight_{uuid.uuid4().hex}"
    assert not scratch_root.exists()

    with pytest.raises(ValueError, match="rust_contract_id"):
        expansion_runner.main(_arguments(scratch_root))
    assert not scratch_root.exists()


def _install_ported_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = expansion_runner.CatalogEntry(
        family="test_family",
        signal="test_rust_signal",
        kernel="range_ratio",
        hypothesis="test-only Rust contract fixture",
        rust_contract_id="research/test_rust_signal/v1",
    )
    monkeypatch.setattr(
        expansion_runner,
        "_normalise_catalogue",
        lambda _module: (Path(__file__), "test_rust_v1", (entry,), {entry.family: [entry.signal]}),
    )
    monkeypatch.setattr(
        expansion_runner,
        "build_signal_specs",
        lambda _cfg: [
            FactorSpec(
                name=entry.signal,
                factor=entry.kernel,
                params={"signal": entry.signal, "family": entry.family},
                rust_contract_id=entry.rust_contract_id,
            )
        ],
    )
    monkeypatch.setattr(expansion_runner, "validate_rust_first_contracts", lambda _specs: None)


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
    _install_ported_catalogue(monkeypatch)

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
    assert len(family_catalog["families"]) == 1
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
    _install_ported_catalogue(monkeypatch)

    def failed_run_factor_batch(cfg: dict, *, paths_cfg: dict) -> Path:
        raise RuntimeError("simulated build failure")

    monkeypatch.setattr(expansion_runner, "run_factor_batch", failed_run_factor_batch)

    with pytest.raises(RuntimeError, match="simulated build failure"):
        expansion_runner.main(_arguments(scratch_root, "--execute"))
    assert not scratch_root.exists()
