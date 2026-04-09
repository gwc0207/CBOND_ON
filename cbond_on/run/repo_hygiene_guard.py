from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import subprocess
import sys


# Files/dirs that should never be tracked by git.
BLOCK_PATTERNS = (
    "node_modules/*",
    "wandb/*",
    "runtime/*",
    "notebook/runtime/*",
    "*.dist-info/*",
    "*.egg-info/*",
)


def _tracked_files(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git ls-files failed")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _violations(paths: list[str]) -> list[str]:
    bad: list[str] = []
    for p in paths:
        normalized = p.replace("\\", "/")
        for pattern in BLOCK_PATTERNS:
            if fnmatch(normalized, pattern):
                bad.append(normalized)
                break
    return bad


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        tracked = _tracked_files(repo_root)
    except Exception as exc:
        print(f"repo hygiene guard: error: {exc}")
        return 1

    bad = _violations(tracked)
    if not bad:
        print("repo hygiene guard: ok")
        return 0

    print("repo hygiene guard: blocked tracked artifacts detected")
    for p in bad[:200]:
        print(f"- {p}")
    if len(bad) > 200:
        print(f"... and {len(bad) - 200} more")
    print("\nFix suggestion:")
    print(
        "  git rm -r --cached node_modules wandb runtime notebook/runtime "
        "&& git rm -r --cached *.dist-info *.egg-info"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

