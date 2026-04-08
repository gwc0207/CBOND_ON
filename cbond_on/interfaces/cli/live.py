from __future__ import annotations

import argparse

from cbond_on.app.usecases.run_live_once import execute as run_live_once


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CBOND_ON live pipeline")
    parser.add_argument("--start", help="start date (YYYY-MM-DD)")
    parser.add_argument("--target", help="target date (YYYY-MM-DD)")
    parser.add_argument("--mode", default="default", help="run mode")
    args = parser.parse_args(argv)

    out_dir = run_live_once(start=args.start, target=args.target, mode=args.mode)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()

