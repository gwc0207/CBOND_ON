from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from cbond_on.domain.factors.storage import FactorStore
from harness.tools import run_catalog_live50_fullchain_dryrun as dryrun


def _write_seed(path: Path, text: str = "seed") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _install_preflight_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    scratch_parent = tmp_path / "research_scratch"
    active_factor = tmp_path / "active_factor"
    active_label = tmp_path / "active_label"
    raw_root = tmp_path / "raw"
    clean_root = tmp_path / "clean"
    raw_root.mkdir()
    clean_root.mkdir()
    days = [date(2026, 8, 20), date(2026, 8, 21)]
    for idx, day in enumerate(days):
        # A preflight only hashes/copies these paths, so a compact arbitrary
        # file is enough and avoids depending on a parquet engine in this test.
        _write_seed(FactorStore(active_factor, panel_name="T1430").day_path(day), f"factor-{idx}")
        _write_seed(active_label / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet", f"label-{idx}")

    state = _write_seed(tmp_path / "source" / "state.csv", "trade_date,x\n2026-08-20,1\n")
    regsim = _write_seed(tmp_path / "source" / "regsim.csv", "trade_date,day_return\n2026-08-20,0.01\n")
    ensemble = _write_seed(tmp_path / "source" / "ensemble.csv", "trade_date,day_return\n2026-08-20,0.01\n")
    hl20 = _write_seed(tmp_path / "source" / "hl20.csv", "trade_date,day_return\n2026-08-20,0.01\n")
    regime = _write_seed(tmp_path / "source" / "regime.csv", "trade_date,benchmark_return\n2026-08-20,0.01\n")

    model_switch = {
        "state_feature_path": str(state),
        "champion": {"model_id": "regsim", "return_path": str(regsim)},
        "challenger": {
            "model_id": "ensemble",
            "kind": "rankavg",
            "score_output": str(tmp_path / "old_scores" / "ensemble"),
            "return_path": str(ensemble),
            "sources": [{"model_id": "baseline", "config": "score/baseline"}],
        },
        "challengers": [
            {
                "model_id": "ensemble",
                "kind": "rankavg",
                "score_output": str(tmp_path / "old_scores" / "ensemble"),
                "return_path": str(ensemble),
                "sources": [{"model_id": "baseline", "config": "score/baseline"}],
            },
            {"model_id": "hl20", "config": "score/hl20", "return_path": str(hl20)},
        ],
    }
    active_live = {
        "runtime": {"paths_config": "data/active_paths"},
        "factor": {"config": "live/factors"},
        "model_switch": model_switch,
    }
    active_paths = {
        "raw_data_root": str(raw_root),
        "cleaned_data_root": str(clean_root),
        "clean_data_root": str(clean_root),
        "panel_data_root": str(tmp_path / "active_panel"),
        "label_data_root": str(active_label),
        "factor_data_root": str(active_factor),
    }
    dryrun_live = {
        "runtime": {"paths_config": "data/old_dryrun_paths"},
        "factor": {"config": "live/factors"},
        "model_score": {"config": "score/regsim", "model_id": "regsim"},
        "model_switch": model_switch,
        "output": {"db_write": False},
    }
    score_cfgs = {
        "score/regsim": {"models": {"regsim": {"model_type": "lgbm", "model_config": "model/regsim"}}},
        "score/baseline": {"models": {"baseline": {"model_type": "lgbm", "model_config": "model/baseline"}}},
        "score/hl20": {"models": {"hl20": {"model_type": "lgbm", "model_config": "model/hl20"}}},
    }
    model_cfgs = {
        key: {
            "model_name": key.rsplit("/", 1)[-1],
            "rolling": {"window_days": 2},
            "incremental": {"warm_start": True},
            "feature_engineering": {"regime": {"enabled": True, "source_path": str(regime)}},
        }
        for key in ("model/regsim", "model/baseline", "model/hl20")
    }
    configs = {dryrun.DRYRUN_LIVE_CONFIG: dryrun_live, **score_cfgs, **model_cfgs}
    monkeypatch.setattr(dryrun, "RESEARCH_SCRATCH_PARENT", scratch_parent)
    monkeypatch.setattr(dryrun, "load_config_file", lambda name: configs[str(name)])
    # Preflight fixture intentionally uses compact noncanonical bytes rather
    # than a full table bundle; emulate the verifier's explicit audit reader.
    monkeypatch.setattr(
        dryrun,
        "build_factor_reader",
        lambda *_args, **_kwargs: FactorStore(active_factor, panel_name="T1430"),
    )
    monkeypatch.setattr(
        dryrun,
        "_load_active_live_inputs",
        lambda: (active_live, active_paths, {"fixture": True}, "live/factors"),
    )
    monkeypatch.setattr(dryrun, "prev_trading_days_from_raw", lambda *_args, **_kwargs: list(days))
    return scratch_parent, tmp_path


def test_preflight_builds_no_db_scratch_config_and_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent, _ = _install_preflight_fixture(monkeypatch, tmp_path)
    root = scratch_parent / "r1"

    plan = dryrun.preflight(score_day="2026-08-25", target_day="2026-08-26", scratch_root=root)

    assert not root.exists()
    assert plan["research_only"] is True
    assert plan["admission"]["release_id"] == dryrun.LIVE50_RELEASE_ID
    assert plan["admission"]["factor_count"] == len(dryrun.LIVE50_COLUMNS)
    assert plan["dynamic_live_config"]["output"]["db_write"] is False
    assert plan["dynamic_live_config"]["runtime"]["paths_config"].startswith(str(root))
    assert plan["dynamic_live_config"]["model_score"]["config"].startswith(str(root))
    # The child paths config is an absolute scratch JSON, not a config/data
    # profile.  It must therefore carry every expanded runtime path directly.
    assert plan["scratch_paths"]["panel_data_root"] == str(tmp_path / "active_panel")
    for key in ("label_data_root", "factor_data_root", "ads_root", "results_root", "model_root", "score_root", "logs_root"):
        assert plan["scratch_paths"][key].startswith(str(root))
    assert plan["scratch_paths"]["read_only_input_roots"]["factor_data_root"].startswith(str(root))
    assert len(plan["seed_requests"]) >= 8
    assert {row["kind"] for row in plan["seed_requests"]} >= {
        "factor_history",
        "label_history",
        "state_feature_path",
        "return_path",
        "regime_source",
    }


def test_preflight_rejects_a_dryrun_config_that_can_write_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent, _ = _install_preflight_fixture(monkeypatch, tmp_path)
    original = dryrun.load_config_file

    def unsafe_config(name: str):
        cfg = original(name)
        if str(name) == dryrun.DRYRUN_LIVE_CONFIG:
            cfg = {**cfg, "output": {"db_write": True}}
        return cfg

    monkeypatch.setattr(dryrun, "load_config_file", unsafe_config)
    with pytest.raises(dryrun.CatalogLive50FullchainDryrunError, match="db_write=false"):
        dryrun.preflight(
            score_day="2026-08-25",
            target_day="2026-08-26",
            scratch_root=scratch_parent / "unsafe",
        )


def test_runner_source_keeps_production_permit_unchanged_and_patches_only_child_runtime() -> None:
    source = Path(dryrun.__file__).read_text(encoding="utf-8")
    assert "issue_live50_factor_store_write_permit" not in source
    assert "configure_live_paths_profile" not in source
    assert "live_runtime.load_config_file = child_load_config_file" in source
    assert "live_runtime.run_factor_build =" in source
    assert "--_child-plan" in source
    assert "--preflight" in source and "--execute" in source
