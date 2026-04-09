from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Violation:
    path: Path
    lineno: int
    rule: str
    detail: str


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        files.append(p)
    return files


def _iter_imports(tree: ast.AST) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.lineno, node.module))
    return imports


def _check_file(path: Path, repo_root: Path) -> list[Violation]:
    rel = path.relative_to(repo_root)
    try:
        src = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(src)
    except Exception as exc:
        return [
            Violation(
                path=rel,
                lineno=1,
                rule="parse_error",
                detail=f"failed to parse python source: {exc}",
            )
        ]

    violations: list[Violation] = []
    imports = _iter_imports(tree)
    rel_posix = rel.as_posix()

    for lineno, module in imports:
        if module.startswith(
            (
                "cbond_on.data",
                "cbond_on.factors",
                "cbond_on.models",
                "cbond_on.backtest",
                "cbond_on.report",
                "cbond_on.model_eval",
                "cbond_on.strategies",
                "cbond_on.factor_batch",
                "cbond_on.live",
            )
        ):
            violations.append(
                Violation(
                    path=rel,
                    lineno=lineno,
                    rule="forbid_legacy_packages",
                    detail=f"legacy package import '{module}' is not allowed",
                )
            )

        if module.startswith("cbond_on.services"):
            violations.append(
                Violation(
                    path=rel,
                    lineno=lineno,
                    rule="forbid_services",
                    detail=f"legacy import '{module}' is not allowed",
                )
            )

        if rel_posix.startswith("cbond_on/interfaces/cli/") and module.startswith(
            "cbond_on.app.usecases"
        ):
            violations.append(
                Violation(
                    path=rel,
                    lineno=lineno,
                    rule="cli_to_pipeline_only",
                    detail=f"CLI should import app.pipelines, not '{module}'",
                )
            )

        if rel_posix.startswith("cbond_on/run/") and module.startswith("cbond_on.app"):
            violations.append(
                Violation(
                    path=rel,
                    lineno=lineno,
                    rule="run_to_interfaces_only",
                    detail=f"run entry should import interfaces only, not '{module}'",
                )
            )

        if rel_posix.startswith("cbond_on/domain/") and (
            module.startswith("cbond_on.infra")
            or module.startswith("cbond_on.interfaces")
            or module.startswith("cbond_on.app")
        ):
            violations.append(
                Violation(
                    path=rel,
                    lineno=lineno,
                    rule="domain_purity",
                    detail=f"domain layer cannot depend on '{module}'",
                )
            )

    return violations


def check_architecture(*, repo_root: Path | None = None) -> list[Violation]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "cbond_on"
    launch_root = repo_root / "liveLaunch"

    files = _iter_python_files(package_root)
    if launch_root.exists():
        files.extend(_iter_python_files(launch_root))

    violations: list[Violation] = []
    for p in files:
        violations.extend(_check_file(p, repo_root))
    return violations


def format_violations(violations: list[Violation]) -> str:
    lines = ["architecture guard: violations detected"]
    for v in violations:
        lines.append(f"- {v.path}:{v.lineno} [{v.rule}] {v.detail}")
    return "\n".join(lines)


def main() -> int:
    violations = check_architecture()
    if not violations:
        print("architecture guard: ok")
        return 0
    print(format_violations(violations))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
