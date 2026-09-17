from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.domain.factors.storage import FactorStore
from harness.tools import verify_catalog_live50_factor_stage as verifier


def _admitted_specs() -> list[FactorSpec]:
    return [FactorSpec(name=column, factor=f"fixture_{index}") for index, column in enumerate(verifier.LIVE50_COLUMNS)]


def _install_isolated_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_root = tmp_path / "raw"
    clean_root = tmp_path / "clean"
    scratch_parent = tmp_path / "research_scratch"
    raw_root.mkdir()
    clean_root.mkdir()
    monkeypatch.setattr(verifier, "RESEARCH_SCRATCH_PARENT", scratch_parent)
    live_cfg = {
        "factor": {"config": "live/test_factors"},
        "runtime": {"paths_config": "data/test_paths"},
        "data_hub": {},
    }
    factor_cfg = {
        "panel_name": "T1430",
        "compute": {"engine": "rust", "execution_policy": "rust_first", "backend": "cpu"},
        "panel_source": {"mode": "clean_direct"},
    }
    paths_cfg = {
        "raw_data_root": str(raw_root),
        "cleaned_data_root": str(clean_root),
        "factor_data_root": str(tmp_path / "production_live_factor_store"),
    }
    panel_cfg = {"panel_name": "T1430", "lead_minutes": 1}
    clean_manifest = tmp_path / "clean_manifest.json"
    clean_manifest.write_text(
        '{"required_profile":"cbond_on_live_t1430","validation":{"passed":true}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        verifier,
        "load_config_file",
        lambda key: {
            "live": live_cfg,
            "live/test_factors": factor_cfg,
            "data/test_paths": paths_cfg,
            "panel": panel_cfg,
        }[key],
    )
    prepare_calls: list[object] = []

    def fake_prepare(cfg, *, specs):
        prepare_calls.append((cfg, specs))
        return SimpleNamespace(
            profile="live50_rust50_20260806",
            release_id=verifier.LIVE50_RELEASE_ID,
            factor_columns=verifier.LIVE50_COLUMNS,
        )

    monkeypatch.setattr(verifier, "build_signal_specs", lambda _cfg: _admitted_specs())
    monkeypatch.setattr(verifier, "prepare_live50_factor_admission", fake_prepare)
    monkeypatch.setattr(verifier, "data_hub_runtime_from_live", lambda *_args, **_kwargs: {"manifest_root": "unused"})
    monkeypatch.setattr(
        verifier,
        "run_publish_status",
        lambda **_: {
            "ready": True,
            "done_exists": True,
            "manifest_run_id_consistent": True,
            "manifest_run_id_complete": True,
            "done_path": str(tmp_path / "publish.done"),
            "active_run_id": "fixture_run",
            "manifests": {"clean": {"path": str(clean_manifest)}},
        },
    )
    return scratch_parent, raw_root, clean_root


def test_preflight_uses_current_live_factor_reference_and_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent, _raw_root, _clean_root = _install_isolated_stage(monkeypatch, tmp_path)
    scratch = scratch_parent / "preflight"

    manifest, path = verifier.verify(score_day="2026-08-25", scratch_root=scratch, execute=False)

    assert path is None
    assert manifest["mode"] == "preflight"
    assert manifest["admission"]["release_id"] == verifier.LIVE50_RELEASE_ID
    assert manifest["admission"]["columns"] == list(verifier.LIVE50_COLUMNS)
    assert not scratch.exists()


def test_execute_writes_exact_ordered_50_columns_only_to_fresh_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent, raw_root, clean_root = _install_isolated_stage(monkeypatch, tmp_path)
    scratch = scratch_parent / "execute"
    day = pd.Timestamp("2026-08-25").date()
    calls: dict[str, object] = {}

    def fake_pipeline(panel_root, factor_root, start, end, **kwargs):
        calls.update({"panel_root": panel_root, "factor_root": factor_root, "start": start, "end": end, **kwargs})
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(day), "110001.SH")], names=["dt", "code"]
        )
        frame = pd.DataFrame({column: [float(index)] for index, column in enumerate(verifier.LIVE50_COLUMNS)}, index=index)
        FactorStore(Path(factor_root), panel_name="T1430").write_day(day, frame)
        return SimpleNamespace(written=1, skipped=0)

    monkeypatch.setattr(verifier, "run_factor_pipeline", fake_pipeline)

    manifest, manifest_path = verifier.verify(score_day=day, scratch_root=scratch, execute=True)

    assert manifest_path is not None and manifest_path.is_file()
    assert manifest["status"] == "completed"
    assert manifest["admission"]["release_id"] == verifier.LIVE50_RELEASE_ID
    assert manifest["output"]["columns"] == list(verifier.LIVE50_COLUMNS)
    assert manifest["output"]["parquet_sha256"]
    assert manifest["output"]["columns_sha256"] == verifier._json_sha256(list(verifier.LIVE50_COLUMNS))
    assert calls["panel_root"] == clean_root
    assert calls["factor_root"] == scratch / "factor_data"
    assert calls["raw_data_root"] == raw_root
    assert calls["cleaned_data_root"] == clean_root
    assert not (tmp_path / "production_live_factor_store").exists()


def test_execute_rejects_nonfresh_or_nonresearch_scratch_before_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent, _raw_root, _clean_root = _install_isolated_stage(monkeypatch, tmp_path)
    existing = scratch_parent / "existing"
    existing.mkdir(parents=True)
    monkeypatch.setattr(
        verifier,
        "run_factor_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    with pytest.raises(FileExistsError, match="fresh scratch root"):
        verifier.verify(score_day="2026-08-25", scratch_root=existing, execute=True)
    with pytest.raises(verifier.Live50StageVerificationError, match="strictly below"):
        verifier.verify(score_day="2026-08-25", scratch_root=tmp_path / "outside", execute=False)


def test_verifier_source_has_no_live_runtime_db_score_trade_or_scheduler_call() -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    forbidden = (
        "from cbond_on.app.usecases.live_runtime import",
        "from cbond_on.app.pipelines.live_pipeline import",
        "write_trades_to_db(",
        "run_model_score(",
        "select_signals(",
        "liveLaunch.scheduler",
        "run_once(",
    )
    assert not any(token in source for token in forbidden)
