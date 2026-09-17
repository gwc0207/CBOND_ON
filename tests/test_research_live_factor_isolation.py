from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from cbond_on.config.loader import load_config_file
from cbond_on.infra.factors.quality import load_factor_specs_from_cfg
from cbond_on.workflows.research import factor_batch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESEARCH_MODULES = {
    "cbond_on.domain.factors.operators.ai_factory_wave80_intraday_v1": {
        "ai_factory_wave80_intraday_v1",
    },
    "cbond_on.domain.factors.operators.daily_prior_intraday_return_surprise_v1": {
        "daily_prior_intraday_return_surprise_v1",
    },
    "cbond_on.domain.factors.operators.daily_prior_intraday_sharpe_v1": {
        "daily_prior_intraday_sharpe_v1",
    },
    "cbond_on.domain.factors.operators.parity_adjusted_stock_lag_v1": {
        "parity_adjusted_stock_lag_v1",
    },
    "cbond_on.domain.factors.operators.parity_adjusted_stock_lag_v2": {
        "parity_adjusted_stock_lag_v2",
    },
    "cbond_on.domain.factors.operators.research_factor_mining_catalog_v1": {
        "factor_mining_cross_asset_catalog_v1",
        "factor_mining_daily_catalog_v1",
        "factor_mining_hybrid_catalog_v1",
        "factor_mining_intraday_catalog_v1",
    },
    "cbond_on.domain.factors.operators.t1430_amount_accel_depth_delta_v2": {
        "t1430_amount_accel_depth_delta_v2",
    },
    "cbond_on.domain.factors.operators.tail_path_efficiency_5m_v1": {
        "tail_path_efficiency_5m_v1",
    },
}
_RESEARCH_KEYS = frozenset().union(*_RESEARCH_MODULES.values())
_RESEARCH_CONFIG_DIR = _REPO_ROOT / "cbond_on" / "config" / "factor" / "research"


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
        line.removeprefix("ISOLATION_PROBE=")
        for line in result.stdout.splitlines()
        if line.startswith("ISOLATION_PROBE=")
    ]
    assert len(payloads) == 1, result.stdout
    return json.loads(payloads[0])


def _research_cfg(*, module: str, factor_key: str) -> dict[str, object]:
    return {
        "start": "2026-04-28",
        "end": "2026-04-29",
        "panel_name": "T1430",
        "factors": [{"name": factor_key, "factor": factor_key}],
        "factor_files": [],
        "research_only": True,
        "research_factor_modules": [module],
    }


def test_fresh_live_factor_runtime_excludes_research_registry_entries() -> None:
    payload = _fresh_probe(
        """
import json

from cbond_on.app.usecases import factor_build_runtime  # noqa: F401
from cbond_on.app.usecases.factor_batch_runtime import build_signal_specs
from cbond_on.config.loader import load_config_file
from cbond_on.core.registry import FactorRegistry

research_keys = {
    "ai_factory_wave80_intraday_v1",
    "daily_prior_intraday_return_surprise_v1",
    "daily_prior_intraday_sharpe_v1",
    "parity_adjusted_stock_lag_v1",
    "parity_adjusted_stock_lag_v2",
    "factor_mining_cross_asset_catalog_v1",
    "factor_mining_daily_catalog_v1",
    "factor_mining_hybrid_catalog_v1",
    "factor_mining_intraday_catalog_v1",
    "t1430_amount_accel_depth_delta_v2",
    "tail_path_efficiency_5m_v1",
}
live_specs = build_signal_specs(load_config_file("live/live_factors_config"))
live_implementations = {spec.factor for spec in live_specs}
registered = set(FactorRegistry.names())
print("ISOLATION_PROBE=" + json.dumps({
    "research_registered": sorted(research_keys.intersection(registered)),
    "research_live_overlap": sorted(research_keys.intersection(live_implementations)),
    "registered_count": len(registered),
    "live_implementation_count": len(live_implementations),
}))
"""
    )

    assert payload["research_registered"] == []
    assert payload["research_live_overlap"] == []
    assert payload["registered_count"] == 0
    assert payload["live_implementation_count"] == 20


def test_explicit_research_loader_registers_only_declared_allowlist_modules() -> None:
    module_paths = sorted(_RESEARCH_MODULES)
    factor_items = [
        {"name": factor_key, "factor": factor_key}
        for factor_key in sorted(_RESEARCH_KEYS)
    ]
    payload = _fresh_probe(
        f"""
import json

from cbond_on.core.registry import FactorRegistry
from cbond_on.workflows.research.factor_batch import load_research_factor_modules

modules = {module_paths!r}
factor_items = {factor_items!r}
loaded = load_research_factor_modules({{
    "start": "2026-04-28",
    "end": "2026-04-29",
    "panel_name": "T1430",
    "factors": factor_items,
    "factor_files": [],
    "research_only": True,
    "research_factor_modules": modules,
}})
expected = {{item["factor"] for item in factor_items}}
registered = set(FactorRegistry.names())
print("ISOLATION_PROBE=" + json.dumps({{
    "loaded": list(loaded),
    "missing": sorted(expected.difference(registered)),
    "unexpected_research": sorted(
        name for name in registered if name.startswith("factor_mining_") and name not in expected
    ),
}}))
"""
    )

    assert payload["loaded"] == module_paths
    assert payload["missing"] == []
    assert payload["unexpected_research"] == []


def test_research_loader_rejects_implicit_or_unallowlisted_imports() -> None:
    module = "cbond_on.domain.factors.operators.daily_prior_intraday_sharpe_v1"
    cfg = _research_cfg(module=module, factor_key="daily_prior_intraday_sharpe_v1")

    cfg["research_only"] = False
    with pytest.raises(ValueError, match="research_only"):
        factor_batch.load_research_factor_modules(cfg)

    cfg = _research_cfg(module=module, factor_key="daily_prior_intraday_sharpe_v1")
    cfg["research_factor_modules"] = []
    with pytest.raises(ValueError, match="declare"):
        factor_batch.load_research_factor_modules(cfg)

    cfg = _research_cfg(module="os", factor_key="daily_prior_intraday_sharpe_v1")
    with pytest.raises(ValueError, match="static allowlist"):
        factor_batch.load_research_factor_modules(cfg)


def test_all_research_factor_configs_declare_their_module_before_execution() -> None:
    factor_to_module = {
        factor_key: module_path
        for module_path, factor_keys in _RESEARCH_MODULES.items()
        for factor_key in factor_keys
    }
    config_paths = sorted(_RESEARCH_CONFIG_DIR.glob("*.json5"))
    config_paths.append(
        _REPO_ROOT
        / "cbond_on"
        / "config"
        / "factor"
        / "ai_factory"
        / "seed"
        / "ai_factor_factory_seed.json5"
    )

    for path in config_paths:
        cfg = load_config_file(str(path))
        specs = load_factor_specs_from_cfg(cfg)
        research_keys = {spec.factor for spec in specs}.intersection(factor_to_module)
        if not research_keys:
            continue
        declared_modules = set(cfg.get("research_factor_modules", []))
        assert cfg.get("research_only") is True, path
        assert {
            factor_to_module[factor_key] for factor_key in research_keys
        }.issubset(declared_modules), path


def test_research_contract_and_catalog_runner_remain_separate_from_live_profile() -> None:
    profile = load_config_file(
        str(_REPO_ROOT / "cbond_on" / "factor_contracts" / "profiles" / "research.json5")
    )
    assert profile.get("research_only") is True
    assert profile.get("factors") == []
    assert "factors_ref" not in profile

    catalog_cfg = load_config_file("factor/research/factor_mining_20260802")
    assert catalog_cfg.get("research_only") is True
    assert "research_factor_modules" not in catalog_cfg
