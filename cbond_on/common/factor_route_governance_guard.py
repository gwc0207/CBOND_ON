"""Static governance guard for CBOND_ON factor-result routes.

The guard does not execute a factor, open a database, start a scheduler, or
modify data. It verifies that normal configurations use one of the three
canonical factor tables and that the remaining direct legacy-storage calls are
limited to a documented migration/audit/no-DB allowlist.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

import json5


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DATA = REPO_ROOT / "cbond_on" / "config" / "data"
POLICY_PATH = REPO_ROOT / "harness" / "policies" / "factor_route_legacy_allowlist.json"
SOURCE_ROOTS = (
    REPO_ROOT / "cbond_on",
    REPO_ROOT / "harness" / "tools",
    REPO_ROOT / "liveLaunch",
)
VALID_TABLE_IDS = {"live", "experiment", "factor_library"}
OFFICIAL_ROOT = "D:/cbond_on/factor_store"
LEGACY_ROOT_TOKENS = (
    "D:/cbond_on/factor_data",
    "D:\\cbond_on\\factor_data",
    "/home/gswzif/cbond_on_runtime/factor_data",
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _load_policy() -> dict[str, Any]:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("factor route allowlist must be an object")
    return payload


def _entry_paths(policy: dict[str, Any], key: str) -> set[str]:
    raw = policy.get(key, [])
    if not isinstance(raw, list):
        raise TypeError(f"allowlist {key} must be a list")
    paths: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError(f"allowlist {key} entries must be objects")
        path = str(item.get("path", "")).strip().replace("\\", "/")
        reason = str(item.get("reason", "")).strip()
        if not path or not reason:
            raise ValueError(f"allowlist {key} entry requires path and reason")
        paths.add(path)
    return paths


def _iter_python_files() -> Iterable[Path]:
    seen: set[Path] = set()
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts or path in seen:
                continue
            seen.add(path)
            yield path


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _config_violations(policy: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    active_writer_paths: list[str] = []
    for path in sorted(CONFIG_DATA.glob("paths*_config.json5")):
        rel = _relative(path)
        payload = json5.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            violations.append(f"{rel}: paths profile is not an object")
            continue
        table = payload.get("factor_table")
        lifecycle = payload.get("lifecycle")
        audit_only = (
            isinstance(lifecycle, dict)
            and lifecycle.get("status") == "audit_only"
            and lifecycle.get("normal_consumer") is False
            and bool(str(lifecycle.get("reason", "")).strip())
        )
        direct_root = "factor_data_root" in payload or (
            isinstance(payload.get("read_only_input_roots"), dict)
            and "factor_data_root" in payload["read_only_input_roots"]
        )
        if table is None:
            if not audit_only:
                violations.append(f"{rel}: noncanonical profile must be explicit audit_only")
            continue
        if audit_only:
            violations.append(f"{rel}: audit_only profile must not declare factor_table")
        if direct_root:
            violations.append(f"{rel}: canonical factor_table profile must not declare direct factor_data_root")
        if not isinstance(table, dict):
            violations.append(f"{rel}: factor_table must be an object")
            continue
        table_id = str(table.get("table_id", "")).strip()
        root = str(table.get("root", "")).strip().replace("\\", "/")
        if table_id not in VALID_TABLE_IDS:
            violations.append(f"{rel}: invalid factor_table.table_id {table_id!r}")
        if root != OFFICIAL_ROOT:
            violations.append(f"{rel}: normal factor_table root must be {OFFICIAL_ROOT}, got {root!r}")
        writer = str(table.get("writer", "")).strip()
        if writer:
            if writer != "admitted_live" or table_id != "live":
                violations.append(f"{rel}: invalid normal writer declaration {writer!r}")
            else:
                active_writer_paths.append(rel)
    expected = "cbond_on/config/data/paths_live50_20260805_config.json5"
    if active_writer_paths != [expected]:
        violations.append(f"admitted_live writer profile must be exactly {expected}, got {active_writer_paths}")
    return violations


def _source_violations(policy: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    factorstore_allow = _entry_paths(policy, "factorstore_constructor_entries")
    writer_allow = _entry_paths(policy, "canonical_writer_constructor_entries")
    canonical_store_allow = _entry_paths(policy, "canonical_store_constructor_entries")
    legacy_root_allow = _entry_paths(policy, "legacy_root_literal_entries")
    for path in _iter_python_files():
        rel = _relative(path)
        text = path.read_text(encoding="utf-8-sig")
        if rel != "cbond_on/common/factor_route_governance_guard.py":
            for token in LEGACY_ROOT_TOKENS:
                if token in text and rel not in legacy_root_allow:
                    violations.append(f"{rel}: legacy root literal {token!r} is not allowlisted")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{rel}: cannot parse for route guard: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "FactorStore" and rel not in factorstore_allow:
                violations.append(f"{rel}:{node.lineno}: FactorStore constructor is not allowlisted")
            if name == "CanonicalFactorTableWriter" and rel not in writer_allow:
                violations.append(f"{rel}:{node.lineno}: canonical writer constructor is not allowlisted")
            if name == "CanonicalFactorStore" and rel not in canonical_store_allow:
                violations.append(f"{rel}:{node.lineno}: canonical store constructor is not allowlisted")
    return violations


def _document_violations() -> list[str]:
    required = {
        REPO_ROOT / "README.md": ("factor_table", "D:/cbond_on/factor_store"),
        REPO_ROOT / "harness" / "README.md": ("factor_table", "FactorStore"),
        REPO_ROOT / "harness" / "context" / "source_of_truth.md": ("factor_table", "factor_library"),
        REPO_ROOT / "docs" / "factor_development.md": ("factor_table", "factor_store"),
    }
    violations: list[str] = []
    for path, tokens in required.items():
        text = path.read_text(encoding="utf-8-sig")
        missing = [token for token in tokens if token not in text]
        if missing:
            violations.append(f"{_relative(path)}: missing route-governance text {missing}")
    return violations


def collect_violations() -> list[str]:
    policy = _load_policy()
    if policy.get("canonical_store_root") != OFFICIAL_ROOT:
        return ["factor route allowlist canonical_store_root drift"]
    return [*_config_violations(policy), *_source_violations(policy), *_document_violations()]


def main() -> int:
    violations = collect_violations()
    if violations:
        print("factor route governance: FAILED")
        for item in violations:
            print("-", item)
        return 1
    print("factor route governance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
