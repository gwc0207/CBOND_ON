from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import harness.tools.r88_factor_backfill as r88
from cbond_on.workflows.research import factor_batch


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
        line.removeprefix("R88_ADMISSION_PROBE=")
        for line in result.stdout.splitlines()
        if line.startswith("R88_ADMISSION_PROBE=")
    ]
    assert len(payloads) == 1, result.stdout
    return json.loads(payloads[0])


def _r88_cfg(*, factor_keys: list[str] | None = None) -> dict[str, object]:
    selected_keys = list(factor_keys or factor_batch.R88_RESEARCH_FACTOR_MODULE_MAP)
    return {
        "research_only": True,
        "research_factor_admission_profile": factor_batch.R88_RESEARCH_ADMISSION_PROFILE,
        "research_factor_modules": list(factor_batch.R88_RESEARCH_FACTOR_MODULES),
        "research_factor_module_map": dict(factor_batch.R88_RESEARCH_FACTOR_MODULE_MAP),
        "factors": [
            {"name": f"r88_{index:02d}", "factor": factor_key}
            for index, factor_key in enumerate(selected_keys)
        ],
        "factor_files": [],
    }


def test_checked_in_r88_profile_has_exact_static_28_module_30_key_admission() -> None:
    _path, profile = r88._load_profile()
    payload, _pending = r88._validate_profile_structure(profile)

    assert tuple(profile["research_factor_modules"]) == factor_batch.R88_RESEARCH_FACTOR_MODULES
    assert profile["research_kernel_modules"] == dict(
        factor_batch.R88_RESEARCH_FACTOR_MODULE_MAP
    )
    assert len(profile["research_factor_modules"]) == 28
    assert len(profile["research_kernel_modules"]) == 30
    assert set(profile["research_kernel_modules"]).issubset(
        {str(item["factor"]) for item in payload}
    )


def test_r88_research_batch_explicitly_registers_all_30_factor_keys() -> None:
    payload = _fresh_probe(
        """
import json

from cbond_on.core.registry import FactorRegistry
from cbond_on.workflows.research.factor_batch import (
    R88_RESEARCH_ADMISSION_PROFILE,
    R88_RESEARCH_FACTOR_MODULE_MAP,
    R88_RESEARCH_FACTOR_MODULES,
    prepare_factor_modules,
)

cfg = {
    "research_only": True,
    "research_factor_admission_profile": R88_RESEARCH_ADMISSION_PROFILE,
    "research_factor_modules": list(R88_RESEARCH_FACTOR_MODULES),
    "research_factor_module_map": dict(R88_RESEARCH_FACTOR_MODULE_MAP),
    "factor_files": [],
    "factors": [
        {"name": f"r88_{index:02d}", "factor": factor_key}
        for index, factor_key in enumerate(R88_RESEARCH_FACTOR_MODULE_MAP)
    ],
}
prepare_factor_modules(cfg)
registered_modules = {
    factor_key: FactorRegistry.get(factor_key).__module__
    for factor_key in R88_RESEARCH_FACTOR_MODULE_MAP
}
print("R88_ADMISSION_PROBE=" + json.dumps({
    "registered_modules": registered_modules,
    "expected_modules": dict(R88_RESEARCH_FACTOR_MODULE_MAP),
    "loaded_modules": cfg["research_factor_modules"],
}))
"""
    )

    assert payload["loaded_modules"] == list(factor_batch.R88_RESEARCH_FACTOR_MODULES)
    assert payload["registered_modules"] == payload["expected_modules"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda cfg: cfg.update(
                {"research_factor_modules": cfg["research_factor_modules"][:-1]}
            ),
            "exactly equal the static R88 allowlist",
        ),
        (
            lambda cfg: cfg.update(
                {"research_factor_modules": [*cfg["research_factor_modules"], "os"]}
            ),
            "exactly equal the static R88 allowlist",
        ),
        (
            lambda cfg: cfg["research_factor_module_map"].update(
                {"factor_mining_daily_catalog_v1": "os"}
            ),
            "exactly equal the static R88 mapping",
        ),
        (
            lambda cfg: cfg.update(
                {"factors": cfg["factors"][:-1]}
            ),
            "missing static research factor key",
        ),
    ],
)
def test_r88_research_batch_rejects_missing_or_unknown_module_admission(
    mutator,
    match: str,
) -> None:
    cfg = _r88_cfg()
    mutator(cfg)

    with pytest.raises(ValueError, match=match):
        factor_batch.load_research_factor_modules(cfg)


def test_fresh_live_runtime_does_not_import_r88_metadata_modules() -> None:
    payload = _fresh_probe(
        """
import json
import sys

from cbond_on.app.usecases import factor_build_runtime  # noqa: F401
from cbond_on.core.registry import FactorRegistry
from cbond_on.workflows.research.factor_batch import R88_RESEARCH_FACTOR_MODULE_MAP

r88_keys = set(R88_RESEARCH_FACTOR_MODULE_MAP)
r88_modules = set(R88_RESEARCH_FACTOR_MODULE_MAP.values())
print("R88_ADMISSION_PROBE=" + json.dumps({
    "registered_r88_keys": sorted(r88_keys.intersection(FactorRegistry.names())),
    "imported_r88_modules": sorted(r88_modules.intersection(sys.modules)),
}))
"""
    )

    assert payload["registered_r88_keys"] == []
    assert payload["imported_r88_modules"] == []
