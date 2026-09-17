"""Remove only generated Python cache directories below the factor source roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = (
    REPO_ROOT / "cbond_on" / "domain" / "factors",
    REPO_ROOT / "factor_engine",
)


def _targets() -> list[Path]:
    targets: list[Path] = []
    for root in ROOTS:
        resolved_root = root.resolve()
        if not resolved_root.is_dir():
            raise RuntimeError(f"factor root missing: {resolved_root}")
        for candidate in resolved_root.rglob("__pycache__"):
            resolved = candidate.resolve()
            if candidate.is_dir() and resolved_root in resolved.parents and resolved.name == "__pycache__":
                targets.append(resolved)
    return sorted(set(targets))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    targets = _targets()
    if args.delete:
        for target in targets:
            shutil.rmtree(target)
        remaining = _targets()
        if remaining:
            raise RuntimeError(f"factor cache cleanup incomplete: {remaining[:5]}")
    print(
        json.dumps(
            {
                "mode": "deleted" if args.delete else "check",
                "cache_directory_count": len(targets),
                "roots": [str(root) for root in ROOTS],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
