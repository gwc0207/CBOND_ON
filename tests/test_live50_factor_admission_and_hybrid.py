from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from cbond_on.app.usecases.factor_batch_runtime import build_signal_specs
from cbond_on.core.config import load_config_file
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.factors import pipeline
from cbond_on.infra.live import config as live_config_module
from cbond_on.infra.live.factor_admission import (
    LIVE50_COLUMNS,
    LIVE50_REGISTRATION_MODULES,
    LIVE50_RUNTIME_METADATA_MODULES,
    LIVE50_RUST50_PROFILE,
    prepare_live50_factor_admission,
    validate_live50_factor_contract_admission,
)
from cbond_on.infra.live.config import load_live_model_runtime, validate_live50_model_score_config


def _active_live50_cfg() -> dict:
    return dict(load_config_file("live/live_factors_50_20260805"))


def _active_live50_specs() -> list[FactorSpec]:
    return build_signal_specs(_active_live50_cfg())


def _frame(index: pd.MultiIndex, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: [float(idx + 1) for idx in range(len(index))] for column in columns}, index=index)


def test_active_live50_config_is_one_rust_first_contract_without_hybrid_fields() -> None:
    live_cfg = load_config_file("live")
    assert live_cfg["factor"]["config"] == "live/live_factors_50_20260805"
    assert str(live_cfg["model_score"]["model_id"]).endswith("_50_20260805")
    cfg = _active_live50_cfg()
    compute = cfg["compute"]
    assert compute["engine"] == "rust"
    assert compute["execution_policy"] == "rust_first"
    assert not {"rust_columns", "python_columns", "preserve_existing_rust_columns"}.intersection(compute)
    assert cfg["live_factor_admission"]["profile"] == LIVE50_RUST50_PROFILE
    assert cfg["live_factor_admission"]["model_feature_contract"] == (
        "models/lgbm/lgbm_live50_feature_contract_20260805"
    )
    specs = _active_live50_specs()
    assert tuple(spec.name for spec in specs) == LIVE50_COLUMNS
    assert len(specs) == 50


def test_live50_admission_rejects_partial_factor_contract() -> None:
    with pytest.raises(ValueError, match="frozen ordered 50-column contract"):
        prepare_live50_factor_admission(_active_live50_cfg(), specs=_active_live50_specs()[-1:])


def test_fresh_factor_build_runtime_does_not_import_nondefault_live50_metadata_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    probe = """
import json
import sys

from cbond_on.app.usecases import factor_build_runtime  # noqa: F401
from cbond_on.infra.live.factor_admission import LIVE50_REGISTRATION_MODULES

modules = [
    module for module in LIVE50_REGISTRATION_MODULES
    if module != 'cbond_on.domain.factors.defs' and module in sys.modules
]
print(json.dumps(sorted(modules)))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout.strip()) == []


def test_live50_contract_profile_is_exact_independent_admission_gate() -> None:
    validate_live50_factor_contract_admission()


def test_live50_admission_loads_one_ordered_rust50_contract() -> None:
    admission = prepare_live50_factor_admission(
        _active_live50_cfg(), specs=_active_live50_specs()
    )
    assert admission is not None
    assert admission.profile == LIVE50_RUST50_PROFILE
    assert admission.factor_columns == LIVE50_COLUMNS
    assert set(admission.modules) == set(LIVE50_RUNTIME_METADATA_MODULES)
    assert admission.feature_contract == "models/lgbm/lgbm_live50_feature_contract_20260805"
    for spec in _active_live50_specs():
        assert FactorRegistry.get(spec.factor).__module__.startswith("cbond_on.domain.factors.defs.")


def test_live50_admission_rejects_module_outside_static_allowlist() -> None:
    cfg = _active_live50_cfg()
    cfg["live_factor_admission"] = dict(cfg["live_factor_admission"])
    cfg["live_factor_admission"]["modules"] = ["os"]
    with pytest.raises(ValueError, match="static live50 registration"):
        prepare_live50_factor_admission(cfg, specs=_active_live50_specs())


def test_live50_admission_rejects_retired_hybrid_configuration() -> None:
    cfg = _active_live50_cfg()
    cfg["compute"] = dict(cfg["compute"])
    cfg["compute"]["python_columns"] = ["any"]
    with pytest.raises(ValueError, match="retired hybrid fields"):
        prepare_live50_factor_admission(cfg, specs=_active_live50_specs())


def test_live50_admission_rejects_any_parameter_drift_via_profile_hash() -> None:
    specs = _active_live50_specs()
    first = specs[0]
    changed_params = dict(first.params)
    changed_params["window"] = int(changed_params["window"]) + 1
    specs[0] = replace(first, params=changed_params)
    with pytest.raises(ValueError, match="spec payload differs from the frozen profile"):
        prepare_live50_factor_admission(_active_live50_cfg(), specs=specs)


def test_live50_admission_rejects_changed_or_duplicate_rust_contract_id() -> None:
    changed = _active_live50_specs()
    changed[0] = replace(changed[0], rust_contract_id="live50_r5/not_the_active_spec")
    with pytest.raises(ValueError, match="rust_contract_id"):
        prepare_live50_factor_admission(_active_live50_cfg(), specs=changed)

    duplicated = _active_live50_specs()
    duplicated[1] = replace(duplicated[1], rust_contract_id=duplicated[0].rust_contract_id)
    with pytest.raises(ValueError, match="must not reuse a rust_contract_id"):
        prepare_live50_factor_admission(_active_live50_cfg(), specs=duplicated)


def test_live_model_and_every_switch_source_use_the_same_ordered_live50_features() -> None:
    live_cfg = dict(load_config_file("live"))
    _model_cfg_key, _model_score_cfg, primary_model_id = load_live_model_runtime(live_cfg)

    switch_cfg = dict(live_cfg["model_switch"])
    sources: list[dict] = [
        {
            "name": "primary",
            "model_id": primary_model_id,
            "config": live_cfg["model_score"]["config"],
        }
    ]
    for challenger in [switch_cfg.get("challenger"), *(switch_cfg.get("challengers") or [])]:
        if not isinstance(challenger, dict):
            continue
        if challenger.get("config"):
            sources.append(challenger)
        for source in challenger.get("sources", []) if isinstance(challenger.get("sources"), list) else []:
            if isinstance(source, dict):
                sources.append(source)

    seen: set[tuple[str, str]] = set()
    for source in sources:
        config_key = str(source.get("config", "")).strip()
        model_id = str(source.get("model_id", "")).strip()
        if not config_key or not model_id or (model_id, config_key) in seen:
            continue
        seen.add((model_id, config_key))
        assert validate_live50_model_score_config(
            dict(load_config_file(config_key)),
            expected_model_id=model_id,
            source=f"test source {config_key}",
        ) == model_id

    assert len(seen) == 4


def test_live_model_feature_admission_rejects_a_legacy_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live_config_module,
        "load_config_file",
        lambda _key: {"factors": list(LIVE50_COLUMNS[:-1])},
    )
    with pytest.raises(ValueError, match="exactly the ordered live50 feature contract"):
        validate_live50_model_score_config(
            {"models": {"test_live50": {"model_config": "live/test"}}},
            expected_model_id="test_live50",
            source="test",
        )


def test_live50_standard_rust_pipeline_uses_single_rust_builder_without_hybrid_symbol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The live admission and factor pipeline use the one standard Rust call."""

    cfg = _active_live50_cfg()
    specs = _active_live50_specs()
    prepare_live50_factor_admission(cfg, specs=specs)

    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-08-06 14:30"), "110001.SH")], names=["dt", "code"]
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "_load_factor_panel",
        lambda *_args, **_kwargs: pipeline._PanelLoadOutcome(
            panel=pd.DataFrame({"last": [1.0]}), elapsed_s=0.0, message="mock"
        ),
    )

    def fake_rust(_panel: pd.DataFrame, routed_specs: list[FactorSpec], **_: object) -> pd.DataFrame:
        calls["rust"] = [spec.name for spec in routed_specs]
        return _frame(index, [spec.name for spec in routed_specs])

    monkeypatch.setattr(pipeline, "build_factor_frame_rust", fake_rust)
    # The hybrid builder is retired rather than merely bypassed.  Keeping this
    # assertion separate from the Rust call below prevents a future import from
    # silently restoring a second normal execution route.
    assert not hasattr(pipeline, "build_factor_frame_rust_python_hybrid")
    outcome = pipeline._build_factor_for_day(
        date(2026, 8, 6),
        panel_data_root=tmp_path / "panel",
        store=FactorStore(tmp_path / "factor", panel_name="T1430"),
        window_minutes=15,
        panel_name="T1430",
        refresh=True,
        overwrite=True,
        specs=specs,
        raw_data_root=None,
        context_cfg={"stock_enabled": False, "map_enabled": False, "daily_enabled": False},
        map_index=None,
        daily_source_specs={},
        daily_source_indexes={},
        factor_workers=1,
        compute_backend_params={"__compute_backend__": cfg["compute"]},
        factor_engine="rust",
        tail_features_cfg=None,
        panel_source=pipeline._PanelSourceRuntime(mode="cached_panel"),
    )

    assert outcome.written == 1
    assert calls == {"rust": list(LIVE50_COLUMNS)}
