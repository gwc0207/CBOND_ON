from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.services.factor.factor_build_service import run as run_factor_build


def main(
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
) -> None:
    factor_cfg = dict(load_config_file("factor"))
    if start:
        factor_cfg["start"] = start
    if end:
        factor_cfg["end"] = end
    if refresh is not None:
        factor_cfg["refresh"] = bool(refresh)
    if overwrite is not None:
        factor_cfg["overwrite"] = bool(overwrite)

    result = run_factor_build(
        start=parse_date(factor_cfg.get("start")),
        end=parse_date(factor_cfg.get("end")),
        refresh=bool(factor_cfg.get("refresh", False)),
        overwrite=bool(factor_cfg.get("overwrite", False)),
        cfg=factor_cfg,
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build factor data only (without factor backtest/report).")
    parser.add_argument("--start", help="override start date, e.g. 2025-12-01")
    parser.add_argument("--end", help="override end date, e.g. 2026-03-30")
    parser.add_argument(
        "--refresh",
        dest="refresh",
        action="store_true",
        help="force refresh mode",
    )
    parser.add_argument(
        "--no-refresh",
        dest="refresh",
        action="store_false",
        help="disable refresh mode",
    )
    parser.set_defaults(refresh=None)
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="force overwrite mode",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="disable overwrite mode",
    )
    parser.set_defaults(overwrite=None)
    args = parser.parse_args()
    main(
        start=args.start,
        end=args.end,
        refresh=args.refresh,
        overwrite=args.overwrite,
    )
