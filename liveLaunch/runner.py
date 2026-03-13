from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.services.live.live_service import run_once


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="liveLaunch runner wrapper")
    parser.add_argument("--start", required=False)
    parser.add_argument("--target", required=False)
    parser.add_argument("--mode", default="runner")
    args = parser.parse_args(argv)
    out_dir = run_once(start=args.start, target=args.target, mode=args.mode)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()

