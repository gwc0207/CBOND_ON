from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cbond_on.core.registry import FactorRegistry, OperatorRegistry


def test_factor_registry_is_the_strict_operator_registry_alias() -> None:
    """Legacy imports must retain object identity, not merely shared methods."""

    assert FactorRegistry is OperatorRegistry
    assert type(FactorRegistry) is type(OperatorRegistry)


def test_fresh_process_alias_registration_is_bidirectional() -> None:
    """Verify alias registration without mutating this test process' registry."""

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )
    probe = """
import json

from cbond_on.core.registry import FactorRegistry, OperatorRegistry, RegistryError

class RegisteredViaFactor:
    pass

class RegisteredViaOperator:
    pass

FactorRegistry.register("compat_registered_via_factor")(RegisteredViaFactor)
OperatorRegistry.register("compat_registered_via_operator")(RegisteredViaOperator)

try:
    OperatorRegistry.register("compat_registered_via_factor")(object)
except RegistryError as exc:
    duplicate_error = str(exc)
else:
    duplicate_error = ""

try:
    FactorRegistry.get("compat_missing")
except RegistryError as exc:
    missing_error = str(exc)
else:
    missing_error = ""

print(json.dumps({
    "same_object": FactorRegistry is OperatorRegistry,
    "legacy_visible_through_operator": (
        OperatorRegistry.get("compat_registered_via_factor") is RegisteredViaFactor
    ),
    "operator_visible_through_legacy": (
        FactorRegistry.get("compat_registered_via_operator") is RegisteredViaOperator
    ),
    "names": sorted(OperatorRegistry.names()),
    "duplicate_error": duplicate_error,
    "missing_error": missing_error,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["same_object"] is True
    assert payload["legacy_visible_through_operator"] is True
    assert payload["operator_visible_through_legacy"] is True
    assert payload["names"] == [
        "compat_registered_via_factor",
        "compat_registered_via_operator",
    ]
    assert "operator" in payload["duplicate_error"]
    assert "operator" in payload["missing_error"]
