from __future__ import annotations

from pathlib import Path

import json5
import pytest

from cbond_on.core.config import load_config_file, resolve_config_file_path
from cbond_on.infra.factors.factor_table_resolution import assert_admitted_live_factor_writer


def test_active_and_default_paths_resolve_only_the_canonical_live_table(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CBOND_ON_PATHS_CONFIG",
        "CBOND_ON_RUNTIME_ROOT",
        "CBOND_ON_PATHS_PROFILE",
        "CBOND_ON_RAW_ROOT",
        "CBOND_ON_CLEAN_ROOT",
        "CBOND_ON_DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    active_live = load_config_file("live")
    active_paths = load_config_file(active_live["runtime"]["paths_config"])
    default_paths = load_config_file("paths")

    assert active_paths["factor_table"]["table_id"] == "live"
    assert active_paths["factor_table"]["writer"] == "admitted_live"
    assert active_paths["factor_data_root"] == "D:/cbond_on/factor_store/live"
    assert default_paths["factor_table"]["table_id"] == "live"
    assert default_paths["factor_data_root"] == "D:/cbond_on/factor_store/live"
    monkeypatch.setenv(
        "CBOND_ON_PATHS_CONFIG",
        str(resolve_config_file_path("data/paths_live50_20260805")),
    )
    assert_admitted_live_factor_writer(active_paths, operation="test active live writer")


def test_supplement_has_no_legacy_live_factor_root_and_publishes_canonical_library() -> None:
    path = resolve_config_file_path("factor/research/factor_supplement_v1")
    payload = json5.loads(path.read_text(encoding="utf-8"))
    inputs = payload["inputs"]
    output = payload["output"]

    assert "live_factor_data_root" not in inputs
    assert inputs["live_factor_table"]["table_id"] == "live"
    assert inputs["live_factor_table"]["root"]["windows"] == "D:/cbond_on/factor_store"
    assert output["canonical_factor_store_root"]["windows"] == "D:/cbond_on/factor_store"


def test_active_live_paths_raw_config_has_no_direct_factor_data_root() -> None:
    path = Path("cbond_on/config/data/paths_live50_20260805_config.json5")
    raw = json5.loads(path.read_text(encoding="utf-8"))
    assert "factor_data_root" not in raw
    assert "factor_data_root" not in raw["read_only_input_roots"]
    assert raw["factor_table"] == {
        "table_id": "live",
        "root": "D:/cbond_on/factor_store",
        "writer": "admitted_live",
    }


def test_historical_live50_model_profiles_use_the_manifest_bound_live_table() -> None:
    names = (
        "paths_cross_section_atomic_20260815_r1",
        "paths_cross_section_atomic_20260815_r1_smoke",
        "paths_cross_section_atomic_20260815_r1_smoke_r2",
        "paths_non_tree_atomic_models_20260812_r1",
        "paths_non_tree_atomic_models_20260812_r2",
        "paths_non_tree_atomic_models_20260812_r2_smoke",
        "paths_non_tree_cross_sectional_models_20260812_r3",
        "paths_non_tree_cross_sectional_models_20260812_r3_smoke",
        "paths_non_tree_cross_sectional_models_20260813_r3_v3",
        "paths_non_tree_cross_sectional_models_20260813_r3_v3_smoke",
        "paths_non_tree_cross_sectional_models_20260813_r3_v4",
        "paths_non_tree_cross_sectional_models_20260813_r3_v4_smoke",
        "paths_non_tree_cross_sectional_models_20260813_r3_v5_resume",
        "paths_non_tree_cross_sectional_models_20260813_r3_v5_resume_smoke",
    )
    for name in names:
        payload = json5.loads(resolve_config_file_path(f"data/{name}").read_text(encoding="utf-8"))
        assert payload["factor_table"] == {"table_id": "live", "root": "D:/cbond_on/factor_store"}
        assert "factor_data_root" not in payload["read_only_input_roots"]


def test_legacy_direct_factor_profiles_are_explicitly_audit_only() -> None:
    names = (
        "paths_hard_similar60_regsim_1429_r1_20260801",
        "paths_live50_dryrun_20260805",
        "paths_factor_catalog_live50_dryrun_20260826",
        "paths_factor_mining_20260802",
        "paths_daily_prior_intraday_return_surprise_120d_oos_ic_20260731",
        "paths_daily_prior_intraday_sharpe_120d_oos_ic_20260731",
        "paths_parity_adjusted_stock_lag_v2_oos_ic_20260731",
        "paths_t1430_amount_accel_depth_delta_v2_oos_ic_20260731",
        "paths_tail_path_efficiency_5m_oos_ic_20260731",
    )
    for name in names:
        payload = json5.loads(resolve_config_file_path(f"data/{name}").read_text(encoding="utf-8"))
        assert payload["lifecycle"]["status"] == "audit_only"
        assert payload["lifecycle"]["normal_consumer"] is False
