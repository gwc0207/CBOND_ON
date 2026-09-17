from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import harness.tools.publish_t1429_panel_store as publisher


DAY = date(2026, 6, 11)


def _panel_cfg(*, lead_minutes: int = 1) -> dict:
    return {
        "panel_name": "T1430",
        "panel_mode": "snapshot_sequence",
        "assets": ["cbond", "stock"],
        "workers": 1,
        "lead_minutes": lead_minutes,
        "schedule": {"windows": [{"start": "14:30", "end": "14:30"}]},
        "snapshot": {"price_field": "last", "drop_no_trade": True},
        "count_points": 5000,
        "max_lookback_days": 4,
        "asset_overrides": {"stock": {"max_lookback_days": 4}},
        "snapshot_columns": None,
    }


def _source_evidence(day: date = DAY) -> dict:
    return {
        "trade_day": day.isoformat(),
        "datahub_run_id": "datahub_run_1",
        "clean_manifest": {
            "path": f"/clean/{day:%Y-%m-%d}.json",
            "sha256": "a" * 64,
            "status": "success",
            "required_profile": "cbond_on_live_t1430",
        },
        "publish_done": {
            "path": f"/publish/{day:%Y-%m-%d}.done",
            "sha256": "b" * 64,
            "ready": True,
        },
        "clean_snapshots": {
            asset: {"path": f"/{asset}/{day:%Y%m%d}.parquet", "bytes": 1, "mtime_ns": 1}
            for asset in publisher.ASSETS
        },
    }


def _synthetic_panel(day: date, *, after_cutoff: bool = False) -> pd.DataFrame:
    logical = pd.Timestamp(f"{day:%Y-%m-%d} 14:30:00")
    cutoff = pd.Timestamp(f"{day:%Y-%m-%d} 14:29:00")
    trade_time = [pd.Timestamp(f"{day:%Y-%m-%d} 09:30:00")] * publisher.COUNT_POINTS
    trade_time[-1] = cutoff + pd.Timedelta(seconds=1) if after_cutoff else cutoff
    index = pd.MultiIndex.from_arrays(
        [
            [logical] * publisher.COUNT_POINTS,
            ["110001.SH"] * publisher.COUNT_POINTS,
            list(range(publisher.COUNT_POINTS)),
        ],
        names=["dt", "code", "seq"],
    )
    return pd.DataFrame({"trade_time": trade_time, "last": 100.0}, index=index)


def _plan(tmp_path: Path) -> publisher.PanelStorePlan:
    panel_cfg = _panel_cfg()
    contract = publisher._contract_from_panel_config(panel_cfg)
    contract_sha = publisher._sha256_bytes(publisher._canonical_json_bytes(contract))
    source = _source_evidence()
    return publisher.PanelStorePlan(
        target_root=tmp_path / "research_scratch" / "panel_smoke",
        target_kind="scratch",
        start=DAY,
        end=DAY,
        days=(DAY,),
        workers=1,
        panel_cfg=panel_cfg,
        contract=contract,
        contract_sha256=contract_sha,
        calendar={
            "source": "synthetic",
            "days_sha256": publisher._days_sha256([DAY]),
            "raw_cbond_snapshot_days": [DAY.isoformat()],
            "clean_cbond_snapshot_days": [DAY.isoformat()],
            "clean_stock_snapshot_days": [DAY.isoformat()],
            "publish_ready_days": [DAY.isoformat()],
        },
        source_by_day={DAY.isoformat(): source},
    )


def test_contract_rejects_any_non_t1429_input() -> None:
    with pytest.raises(publisher.PanelStorePublisherError, match="lead_minutes"):
        publisher._contract_from_panel_config(_panel_cfg(lead_minutes=0))

    wrong_assets = _panel_cfg()
    wrong_assets["assets"] = ["cbond"]
    with pytest.raises(publisher.PanelStorePublisherError, match="assets"):
        publisher._contract_from_panel_config(wrong_assets)


def test_preflight_is_no_write_and_requires_three_way_calendar_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    public_root = tmp_path / "public_panel_store"
    monkeypatch.setattr(publisher, "PUBLIC_PANEL_ROOT", public_root)
    monkeypatch.setattr(publisher, "DATAHUB_RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(publisher, "DATAHUB_CLEAN_ROOT", tmp_path / "clean")
    monkeypatch.setattr(publisher, "DATAHUB_MANIFEST_ROOT", tmp_path / "manifests")
    monkeypatch.setattr(publisher, "load_config_file", lambda _name: _panel_cfg())
    monkeypatch.setattr(publisher, "list_trading_days_from_raw", lambda *_args, **_kwargs: [DAY])
    monkeypatch.setattr(
        publisher,
        "_iter_existing_snapshot_days",
        lambda *_args, **_kwargs: [DAY],
    )
    monkeypatch.setattr(publisher, "_datahub_day_evidence", lambda day: _source_evidence(day))

    plan = publisher.preflight(start_text=DAY.isoformat(), end_text=DAY.isoformat())

    assert plan.target_root == public_root.resolve(strict=False)
    assert plan.target_kind == "public"
    assert plan.days == (DAY,)
    assert not public_root.exists()

    monkeypatch.setattr(
        publisher,
        "_iter_existing_snapshot_days",
        lambda _root, *_args, asset, **_kwargs: [DAY] if asset == "cbond" else [],
    )
    with pytest.raises(publisher.PanelStorePublisherError, match="clean stock snapshot calendars differ"):
        publisher.preflight(start_text=DAY.isoformat(), end_text=DAY.isoformat())


def test_execute_scope_rejects_ambiguous_or_unapproved_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(publisher, "PUBLIC_PANEL_ROOT", tmp_path / "public")
    monkeypatch.setattr(publisher, "RESEARCH_SCRATCH_PARENT", tmp_path / "research_scratch")

    with pytest.raises(publisher.PanelStorePublisherError, match="requires exactly one"):
        publisher._resolve_target(execute=True, public_root=False, scratch_root=None)
    with pytest.raises(publisher.PanelStorePublisherError, match="must resolve strictly below"):
        publisher._resolve_target(execute=True, public_root=False, scratch_root=str(tmp_path / "outside"))

    target, kind = publisher._resolve_target(
        execute=True,
        public_root=False,
        scratch_root=str(tmp_path / "research_scratch" / "smoke"),
    )
    assert kind == "scratch"
    assert target == (tmp_path / "research_scratch" / "smoke").resolve(strict=False)


def test_scratch_preflight_is_bounded_to_a_smoke_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scratch_parent = tmp_path / "research_scratch"
    days = [date(2026, 6, 1 + offset) for offset in range(publisher.SCRATCH_SMOKE_MAX_DAYS + 1)]
    monkeypatch.setattr(publisher, "RESEARCH_SCRATCH_PARENT", scratch_parent)
    monkeypatch.setattr(publisher, "load_config_file", lambda _name: _panel_cfg())
    monkeypatch.setattr(publisher, "list_trading_days_from_raw", lambda *_args, **_kwargs: days)
    monkeypatch.setattr(publisher, "_iter_existing_snapshot_days", lambda *_args, **_kwargs: days)
    monkeypatch.setattr(publisher, "_datahub_day_evidence", lambda day: _source_evidence(day))

    with pytest.raises(publisher.PanelStorePublisherError, match="smoke-only"):
        publisher.preflight(
            start_text=days[0].isoformat(),
            end_text=days[-1].isoformat(),
            execute=True,
            scratch_root=str(scratch_parent / "too_many_days"),
        )


def test_execute_publishes_two_assets_then_atomic_manifest_and_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(publisher, "RESEARCH_SCRATCH_PARENT", tmp_path / "research_scratch")
    plan = _plan(tmp_path)
    assert not plan.target_root.exists()
    monkeypatch.setattr(publisher, "_datahub_day_evidence", lambda day: _source_evidence(day))

    def fake_build(_clean, stage_root, _raw, *_args, asset: str, **_kwargs):
        path = publisher._panel_path(Path(stage_root), DAY, asset)
        path.parent.mkdir(parents=True, exist_ok=True)
        _synthetic_panel(DAY).to_parquet(path)
        return SimpleNamespace(written=1, skipped=0, missing_snapshot_days=0)

    writes: list[str] = []
    original_write_json = publisher._write_json_atomic

    def tracked_write(path: Path, payload: dict) -> None:
        writes.append(path.name)
        original_write_json(path, payload)

    monkeypatch.setattr(publisher, "build_panel_data", fake_build)
    monkeypatch.setattr(publisher, "_write_json_atomic", tracked_write)

    result = publisher.execute(plan)

    assert result["days_published"] == 1
    assert not (plan.target_root / ".staging").exists()
    manifest_path = publisher._day_manifest_path(plan.target_root, DAY)
    done_path = publisher._day_done_path(plan.target_root, DAY)
    assert manifest_path.is_file()
    assert done_path.is_file()
    assert writes.index(manifest_path.name) < writes.index(done_path.name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "published"
    assert manifest["contract_sha256"] == plan.contract_sha256
    assert set(manifest["assets"]) == set(publisher.ASSETS)
    assert done["ready"] is True
    assert done["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert Path(result["batch_done"]).is_file()
    for asset in publisher.ASSETS:
        path = publisher._panel_path(plan.target_root, DAY, asset)
        assert path.is_file()
        assert done["assets"][asset] == manifest["assets"][asset]["sha256"]


def test_panel_verification_rejects_a_tick_after_the_t1429_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "after_cutoff.parquet"
    _synthetic_panel(DAY, after_cutoff=True).to_parquet(path)

    with pytest.raises(publisher.PanelStorePublisherError, match="physical cutoff"):
        publisher._verify_panel_file(path, day=DAY, asset="cbond")


def test_failed_done_commit_leaves_no_visible_day_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(publisher, "RESEARCH_SCRATCH_PARENT", tmp_path / "research_scratch")
    plan = _plan(tmp_path)
    monkeypatch.setattr(publisher, "_datahub_day_evidence", lambda day: _source_evidence(day))

    def fake_build(_clean, stage_root, _raw, *_args, asset: str, **_kwargs):
        path = publisher._panel_path(Path(stage_root), DAY, asset)
        path.parent.mkdir(parents=True, exist_ok=True)
        _synthetic_panel(DAY).to_parquet(path)
        return SimpleNamespace(written=1, skipped=0, missing_snapshot_days=0)

    original_write_json = publisher._write_json_atomic

    def fail_only_day_done(path: Path, payload: dict) -> None:
        if path == publisher._day_done_path(plan.target_root, DAY):
            raise OSError("synthetic done failure")
        original_write_json(path, payload)

    monkeypatch.setattr(publisher, "build_panel_data", fake_build)
    monkeypatch.setattr(publisher, "_write_json_atomic", fail_only_day_done)

    with pytest.raises(OSError, match="synthetic done failure"):
        publisher.execute(plan)

    assert not publisher._day_done_path(plan.target_root, DAY).exists()
    assert not publisher._day_manifest_path(plan.target_root, DAY).exists()
    assert not publisher._panel_path(plan.target_root, DAY, "cbond").exists()
    assert not publisher._panel_path(plan.target_root, DAY, "stock").exists()
    assert not (plan.target_root / ".staging").exists()


def test_publisher_source_has_no_live_factor_model_or_db_execution_dependency() -> None:
    source = Path(publisher.__file__).read_text(encoding="utf-8")
    forbidden = (
        "run_factor_pipeline(",
        "run_model_score(",
        "write_trades_to_db(",
        "liveLaunch.scheduler",
        "FactorStore(",
    )
    assert not [item for item in forbidden if item in source]
