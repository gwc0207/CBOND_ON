from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "cbond_on"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _assert_no_forbidden_imports(root: Path, forbidden_prefixes: tuple[str, ...]) -> None:
    violations: list[str] = []
    for path in _python_files(root):
        for imported in sorted(_imports(path)):
            if imported.startswith(forbidden_prefixes):
                rel = path.relative_to(PROJECT_ROOT).as_posix()
                violations.append(f"{rel}: {imported}")
    assert not violations, "Forbidden layer imports:\n" + "\n".join(violations)


def test_domain_does_not_depend_on_outer_layers() -> None:
    _assert_no_forbidden_imports(
        PACKAGE_ROOT / "domain",
        (
            "cbond_on.app",
            "cbond_on.bootstrap",
            "cbond_on.cli",
            "cbond_on.infra",
            "cbond_on.interfaces",
            "cbond_on.workflows",
        ),
    )


def test_ports_do_not_depend_on_implementations() -> None:
    _assert_no_forbidden_imports(
        PACKAGE_ROOT / "ports",
        (
            "cbond_on.app",
            "cbond_on.bootstrap",
            "cbond_on.cli",
            "cbond_on.infra",
            "cbond_on.interfaces",
            "cbond_on.workflows",
        ),
    )


def test_workflows_do_not_depend_on_infra_or_cli() -> None:
    _assert_no_forbidden_imports(
        PACKAGE_ROOT / "workflows",
        (
            "cbond_on.bootstrap",
            "cbond_on.cli",
            "cbond_on.infra",
            "cbond_on.interfaces",
        ),
    )
