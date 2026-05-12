from __future__ import annotations

import argparse

from cbond_on.bootstrap.production import build_live_request
from cbond_on.workflows.production.live import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CBOND_ON live pipeline")
    parser.add_argument("--start", help="start date (YYYY-MM-DD)")
    parser.add_argument("--target", help="target date (YYYY-MM-DD)")
    parser.add_argument("--mode", default="default", help="run mode")
    args = parser.parse_args(argv)

    request = build_live_request(start=args.start, target=args.target, mode=args.mode)
    out_dir = run(start=request.start, target=request.target, mode=request.mode)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()

