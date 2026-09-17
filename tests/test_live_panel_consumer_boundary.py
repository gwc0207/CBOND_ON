from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from cbond_on.app.usecases import live_runtime
from cbond_on.core.config import load_config_file
from cbond_on.infra.factors import pipeline
from harness.tools import run_catalog_live50_fullchain_dryrun as dryrun


SCORE_DAY = date(2026, 8, 25)
PREV_DAY = date(2026, 8, 22)
TARGET_DAY = date(2026, 8, 26)


def _paths(tmp_path: Path) -> dict[str, str]:
    return {
        "raw_data_root": str(tmp_path / "raw"),
        "clean_data_root": str(tmp_path / "clean"),
        "cleaned_data_root": str(tmp_path / "clean"),
        "panel_data_root": str(tmp_path / "published_panel_store"),
        "label_data_root": str(tmp_path / "label_store"),
        "factor_data_root": str(tmp_path / "factor_store"),
        "results_root": str(tmp_path / "results"),
    }


def _write_panel(paths_cfg: dict[str, str], day: date, asset: str) -> None:
    path = live_runtime._panel_day_path(paths_cfg, day, asset=asset, panel_name="T1430")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"published panel")


def _write_published_bundle(paths_cfg: dict[str, str], day: date) -> None:
    assets: dict[str, dict[str, str]] = {}
    for asset in ("cbond", "stock"):
        _write_panel(paths_cfg, day, asset)
        path = live_runtime._panel_day_path(paths_cfg, day, asset=asset, panel_name="T1430")
        assets[asset] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest_path = live_runtime._panel_manifest_day_path(paths_cfg, day, panel_name="T1430")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"status": "published", "trade_day": day.isoformat(), "assets": assets}),
        encoding="utf-8",
    )
    done_path = live_runtime._panel_done_day_path(paths_cfg, day, panel_name="T1430")
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(
        json.dumps(
            {
                "ready": True,
                "trade_day": day.isoformat(),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "assets": {asset: row["sha256"] for asset, row in assets.items()},
            }
        ),
        encoding="utf-8",
    )


def _candidate_factor_cfg() -> dict:
    return {
        "panel_name": "T1430",
        "panel_source": {"mode": "cached_panel"},
        "panel_store": {"mode": "published_read_only"},
    }


def test_normal_factor_panel_source_defaults_to_clean_direct() -> None:
    default_cfg = load_config_file("factor/runtime/default.json5")
    assert default_cfg["panel_source"]["mode"] == "clean_direct"
    assert pipeline._normalize_panel_source_mode(None) == "clean_direct"
    assert pipeline._normalize_panel_source_mode("") == "clean_direct"
    assert pipeline._normalize_panel_source_mode("cached_panel") == "cached_panel"
    assert live_runtime._factor_panel_source_mode({}) == "clean_direct"
    assert live_runtime._factor_panel_source_mode({"panel_source": {"mode": "cached_panel"}}) == "cached_panel"
    with pytest.raises(ValueError, match="unsupported"):
        live_runtime._factor_panel_source_mode({"panel_source": {"mode": "unknown"}})


def test_cached_panel_coverage_requires_each_configured_asset(tmp_path: Path) -> None:
    paths_cfg = _paths(tmp_path)
    _write_panel(paths_cfg, SCORE_DAY, "cbond")

    with pytest.raises(RuntimeError, match=r"cached PanelStore coverage missing: .*stock"):
        live_runtime._require_cached_panel_coverage(
            paths_cfg=paths_cfg,
            days=[SCORE_DAY],
            assets=["cbond", "stock"],
            panel_name="T1430",
            scope="test",
        )

    assert not (Path(paths_cfg["panel_data_root"]) / "neutralization_cache").exists()


def test_published_panel_store_run_never_calls_panel_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths_cfg = _paths(tmp_path)
    for day in (PREV_DAY, SCORE_DAY):
        _write_published_bundle(paths_cfg, day)

    factor_cfg = _candidate_factor_cfg()
    live_cfg = {
        "schedule": {},
        "model_score": {},
        "strategy": {"strategy_id": "fixture"},
        "output": {"db_write": False},
        "allowlist": {
            "enabled": True,
            "table": live_runtime.O005_ALLOWLIST_TABLE,
        },
    }
    panel_cfg = {"panel_name": "T1430", "assets": ["cbond", "stock"]}
    label_cfg = {"workers": 1}
    model_score_cfg = {"models": {"fixture_model": {"model_config": "fixture/model"}}}
    configs = {
        "live": live_cfg,
        "paths": paths_cfg,
        "panel": panel_cfg,
        "label": label_cfg,
    }
    calls: dict[str, list[dict]] = {"panel": [], "label": [], "factor": []}

    def fake_label_build(**kwargs: object) -> dict:
        calls["label"].append(dict(kwargs))
        for day in {kwargs["start"], kwargs["end"]}:
            path = live_runtime._label_day_path(paths_cfg, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"label")
        return {"written": 1}

    def fake_factor_build(**kwargs: object) -> dict:
        calls["factor"].append(dict(kwargs))
        for day in {kwargs["start"], kwargs["end"]}:
            path = live_runtime._factor_day_path(paths_cfg, day, panel_name="T1430")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"factor")
        return {"written": 1}

    monkeypatch.setattr(live_runtime, "load_config_file", lambda name: configs[str(name)])
    monkeypatch.setattr(live_runtime, "configure_live_paths_profile", lambda _cfg: None)
    monkeypatch.setattr(live_runtime, "assert_admitted_live_factor_writer", lambda *_args, **_kwargs: None)

    class FakeFactorReader:
        def day_path(self, day: date) -> Path:
            return (
                Path(paths_cfg["factor_data_root"])
                / "factors"
                / "T1430"
                / f"{day:%Y-%m}"
                / f"{day:%Y%m%d}.parquet"
            )

        def read_day(self, day: date) -> pd.DataFrame:
            return pd.DataFrame({"f1": [1.0]}) if self.day_path(day).is_file() else pd.DataFrame()

    monkeypatch.setattr(live_runtime, "build_factor_reader", lambda *_args, **_kwargs: FakeFactorReader())
    monkeypatch.setattr(live_runtime, "load_live_factor_runtime", lambda _cfg: ("candidate", factor_cfg))
    monkeypatch.setattr(
        live_runtime,
        "load_live_model_runtime",
        lambda _cfg: ("candidate_models", model_score_cfg, "fixture_model"),
    )
    monkeypatch.setattr(live_runtime, "today_shanghai", lambda: SCORE_DAY)
    monkeypatch.setattr(live_runtime, "_prev_trading_day", lambda *_args, **_kwargs: PREV_DAY)
    monkeypatch.setattr(live_runtime, "prev_trading_days_from_raw", lambda *_args, **_kwargs: [PREV_DAY])
    monkeypatch.setattr(live_runtime, "_parse_live_model_window_days", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        live_runtime,
        "data_hub_runtime_from_live",
        lambda *_args, **_kwargs: {"ready_gate_enabled": True},
    )
    monkeypatch.setattr(live_runtime, "ensure_publish_ready", lambda **_kwargs: None)
    monkeypatch.setattr(live_runtime, "run_label_build", fake_label_build)
    monkeypatch.setattr(live_runtime, "run_factor_build", fake_factor_build)
    monkeypatch.setattr(live_runtime, "run_panel_build", lambda **kwargs: calls["panel"].append(dict(kwargs)))
    monkeypatch.setattr(live_runtime, "run_model_score", lambda **_kwargs: {"score_output": str(tmp_path / "scores")})
    monkeypatch.setattr(
        live_runtime,
        "_score_df_from_path",
        lambda *_args, **_kwargs: pd.DataFrame({"code": ["110001.SH"], "score": [0.5]}),
    )
    monkeypatch.setattr(live_runtime, "load_strategy_config", lambda _path: {})
    monkeypatch.setattr(live_runtime, "read_clean_daily", lambda *_args: pd.DataFrame({"code": ["110001.SH"]}))
    monkeypatch.setattr(live_runtime, "load_upstream_pool_config", lambda _cfg: {})
    monkeypatch.setattr(
        live_runtime,
        "resolve_pool_codes_for_trade_day",
        lambda **_kwargs: (["110001.SH"], {"fallback_no_filter": False}),
    )
    monkeypatch.setattr(live_runtime, "apply_allowlist_filter_to_universe", lambda universe, **_kwargs: universe.copy())
    monkeypatch.setattr(live_runtime, "load_previous_holdings", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(
        live_runtime,
        "select_signals",
        lambda request: pd.DataFrame(
            {"code": ["110001.SH"], "score": [0.5], "weight": [1.0], "rank": [1]}
        ),
    )

    out_dir = live_runtime.run_once(start=SCORE_DAY, target=TARGET_DAY)

    assert out_dir == Path(paths_cfg["results_root"]) / "live" / "2026-08-26"
    assert (out_dir / "trade_list.csv").is_file()
    assert calls["panel"] == []
    assert len(calls["label"]) == 2
    assert len(calls["factor"]) == 2
    assert all(call["cfg"]["panel_source"]["mode"] == "cached_panel" for call in calls["factor"])


def test_retired_panel_candidate_is_unselected_and_clean_direct() -> None:
    active = load_config_file("live")
    candidate = load_config_file("live/live_panel_store_candidate_20260827")
    factor_cfg = load_config_file("live/live_factors_50_panel_store_candidate_20260827")

    assert active["factor"]["config"] != "live/live_factors_50_panel_store_candidate_20260827"
    assert candidate["candidate"] == {
        "mode": "disabled",
        "scheduler_selectable": False,
    }
    assert candidate["output"]["db_write"] is False
    assert factor_cfg["panel_source"]["mode"] == "clean_direct"
    assert "panel_store" not in factor_cfg

    refs = [candidate["model_score"]["config"]]
    for challenger in candidate["model_switch"]["challengers"]:
        if challenger.get("config"):
            refs.append(challenger["config"])
        refs.extend(source["config"] for source in challenger.get("sources", []))
    for score_ref in dict.fromkeys(refs):
        score_cfg = load_config_file(score_ref)
        model_entry = next(iter(score_cfg["models"].values()))
        model_cfg = load_config_file(model_entry["model_config"])
        cache_root = model_cfg["neutralization_cache_root"]
        assert cache_root["windows"].startswith("D:/cbond_on/results/neutralization_cache/live/")
        assert "panel_data" not in cache_root["windows"]


def test_candidate_harness_validation_requires_nonactive_marked_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        "candidate": {"mode": "published_panel_store", "scheduler_selectable": False},
        "output": {"db_write": False},
        "factor": {"config": "live/factor_candidate"},
    }
    factor_cfg = {
        "panel_source": {"mode": "cached_panel"},
        "panel_store": {"mode": "published_read_only"},
    }
    monkeypatch.setattr(
        dryrun,
        "load_config_file",
        lambda key: factor_cfg if str(key) == "live/factor_candidate" else candidate,
    )
    validated: list[str] = []
    monkeypatch.setattr(
        dryrun,
        "_validate_catalog_live50_factor_config",
        lambda _cfg, *, source: validated.append(source),
    )

    assert dryrun._validate_explicit_candidate_live_config("live/candidate", candidate) == "live/factor_candidate"
    assert validated == ["candidate live factor config"]
    with pytest.raises(dryrun.CatalogLive50FullchainDryrunError, match="non-active candidate"):
        dryrun._validate_explicit_candidate_live_config("live", candidate)


def test_disabled_candidate_cannot_enter_dryrun_before_scratch_creation() -> None:
    candidate = {
        "candidate": {"mode": "disabled", "scheduler_selectable": False},
        "output": {"db_write": False},
        "factor": {"config": "live/factor_candidate"},
    }

    with pytest.raises(dryrun.CatalogLive50FullchainDryrunError, match="disabled candidate"):
        dryrun._validate_explicit_candidate_live_config("live/candidate", candidate)


def test_dryrun_cli_forwards_an_explicit_candidate_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        dryrun,
        "preflight",
        lambda **kwargs: calls.append(kwargs) or {"mode": "preflight"},
    )
    monkeypatch.setattr(dryrun, "compact_summary", lambda manifest: manifest)

    status = dryrun.main(
        [
            "--score-day",
            "2026-08-25",
            "--target-day",
            "2026-08-26",
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--live-config",
            "live/live_panel_store_candidate_20260827",
            "--preflight",
        ]
    )

    assert status == 0
    assert calls == [
        {
            "score_day": "2026-08-25",
            "target_day": "2026-08-26",
            "scratch_root": str(tmp_path / "scratch"),
            "live_config": "live/live_panel_store_candidate_20260827",
        }
    ]
