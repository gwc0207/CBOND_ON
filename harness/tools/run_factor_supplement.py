"""CLI for the isolated research-only factor supplement V1.

This tool defaults to ``dry-run``. ``dry-run`` writes only a research ledger;
``execute`` is a one-score-day, catalog-permit-bound research stage that uses
ephemeral scratch staging and publishes only the canonical factor-library
table plus the runtime audit ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import parse_date  # noqa: E402
from cbond_on.workflows.research.factor_supplement import (  # noqa: E402
    DEFAULT_CONFIG_REF,
    compact_summary,
    load_supplement_config,
    run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_REF, help="research-only supplement config")
    parser.add_argument("--score-day", help="YYYY-MM-DD; defaults to local calendar day")
    parser.add_argument("--scratch-root", help="validated child of D:/cbond_on/research_scratch")
    parser.add_argument("--mode", choices=("plan", "dry-run", "execute"), default="dry-run")
    parser.add_argument("--print-full-plan", action="store_true")
    args = parser.parse_args(argv)

    _config_path, config = load_supplement_config(args.config)
    score_day = parse_date(args.score_day) if args.score_day else __import__("datetime").date.today()
    plan, ledger_path = run(
        config,
        score_day=score_day,
        mode=args.mode,
        scratch_root=args.scratch_root,
    )
    output: object = plan if args.print_full_plan else compact_summary(plan, ledger_path=ledger_path)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    if args.mode == "dry-run" and str(plan.get("disposition", "")) != "PLAN_READY_NO_EXECUTION":
        return 2
    if args.mode == "execute" and str(plan.get("disposition", "")) not in {
        "EXECUTION_COMPLETED",
        "EXECUTION_COMPLETED_WITH_COVERAGE_GAPS",
    }:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
