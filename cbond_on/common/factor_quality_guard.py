from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.infra.factors.quality import (
    cleanup_factor_store_columns,
    run_factor_quality_scan,
    update_disabled_factors_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Factor quality guard: detect deprecated factors, bad factors, and "
            "coverage/missing patterns. Default applies fixes (disable-bad + remove-deprecated)."
        )
    )
    parser.add_argument("--config", default="factor", help="factor config key/path (default: factor)")
    parser.add_argument("--start", help="override start day (YYYY-MM-DD)")
    parser.add_argument("--end", help="override end day (YYYY-MM-DD)")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.95,
        help="mark bad if present_days/trading_days below this threshold (default: 0.95)",
    )
    parser.add_argument(
        "--non-null-threshold",
        type=float,
        default=0.80,
        help="mark bad if avg_non_null_ratio below this threshold (default: 0.80)",
    )
    parser.add_argument(
        "--constant-day-threshold",
        type=float,
        default=0.50,
        help="mark bad if constant_day_ratio above this threshold (default: 0.50)",
    )
    parser.add_argument(
        "--min-days-for-coverage-check",
        type=int,
        default=60,
        help="only apply low_coverage rule when trading_days >= this value (default: 60)",
    )
    parser.add_argument("--out", help="optional JSON output path")
    parser.add_argument(
        "--apply-remove-deprecated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "apply action: remove deprecated factor columns from factor parquet store "
            "(default: enabled; use --no-apply-remove-deprecated to disable)"
        ),
    )
    parser.add_argument(
        "--apply-disable-bad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "apply action: add bad factors into disabled_factors file "
            "(default: enabled; use --no-apply-disable-bad to disable)"
        ),
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=50,
        help="max rows to print for deprecated/bad factors (default: 50)",
    )
    return parser.parse_args()


def _resolve_window(cfg: dict, *, start_arg: str | None, end_arg: str | None) -> tuple[date, date]:
    start = parse_date(start_arg) if start_arg else parse_date(cfg.get("start"))
    end = parse_date(end_arg) if end_arg else parse_date(cfg.get("end"))
    if end < start:
        raise ValueError("end must be >= start")
    return start, end


def _print_summary(result: dict, *, show_top: int) -> None:
    print(
        "factor quality guard:",
        f"panel={result.get('panel_label')}",
        f"window={result.get('start')}~{result.get('end')}",
        f"trading_days={result.get('trading_days')}",
        f"expected_factors={result.get('expected_factor_count')}",
    )

    deprecated = list(result.get("deprecated_factors", []))
    bad = list(result.get("bad_factors", []))
    print(f"deprecated_factors={len(deprecated)} bad_factors={len(bad)}")

    if deprecated:
        print("deprecated factor list:")
        for row in deprecated[:show_top]:
            print(
                "-",
                row.get("factor"),
                f"present_days={row.get('present_days')}",
                f"range={row.get('first_present_day')}~{row.get('last_present_day')}",
            )
        if len(deprecated) > show_top:
            print(f"... and {len(deprecated) - show_top} more deprecated factors")

    if bad:
        print("bad factor list:")
        for row in bad[:show_top]:
            print(
                "-",
                row.get("factor"),
                f"reasons={','.join(row.get('bad_reasons', []))}",
                f"missing_ratio={row.get('missing_ratio')}",
                f"null_ratio={row.get('null_ratio')}",
                f"constant_day_ratio={row.get('constant_day_ratio')}",
                f"present_days={row.get('present_days')}",
            )
        if len(bad) > show_top:
            print(f"... and {len(bad) - show_top} more bad factors")


def _print_list(title: str, values: list[str], *, show_top: int) -> None:
    if not values:
        print(f"{title}: []")
        return
    print(f"{title} (count={len(values)}):")
    for x in values[:show_top]:
        print("-", x)
    if len(values) > show_top:
        print(f"... and {len(values) - show_top} more")


def main() -> int:
    args = _parse_args()
    factor_cfg = load_config_file(args.config)
    paths_cfg = load_config_file("paths")
    start, end = _resolve_window(factor_cfg, start_arg=args.start, end_arg=args.end)

    result = run_factor_quality_scan(
        factor_cfg=factor_cfg,
        paths_cfg=paths_cfg,
        start=start,
        end=end,
        coverage_threshold=float(args.coverage_threshold),
        non_null_threshold=float(args.non_null_threshold),
        constant_day_threshold=float(args.constant_day_threshold),
        min_days_for_coverage_check=max(1, int(args.min_days_for_coverage_check)),
    )
    _print_summary(result, show_top=max(1, int(args.show_top)))

    actions: dict[str, object] = {}
    bad_factor_names = [str(row.get("factor", "")).strip() for row in result.get("bad_factors", [])]
    bad_factor_names = [x for x in bad_factor_names if x]
    deprecated_names = [
        str(row.get("factor", "")).strip() for row in result.get("deprecated_factors", [])
    ]
    deprecated_names = [x for x in deprecated_names if x]

    if bool(args.apply_disable_bad):
        disable_update = update_disabled_factors_file(
            factor_cfg,
            add_names=bad_factor_names,
        )
        actions["disable_bad"] = disable_update
        print(
            "applied disable_bad:",
            f"path={disable_update.get('path')}",
            f"after_count={disable_update.get('after_count')}",
            f"added_count={disable_update.get('added_count')}",
        )
        _print_list(
            "disable_bad added_factors",
            list(disable_update.get("added_factors", [])),
            show_top=max(1, int(args.show_top)),
        )
        _print_list(
            "disable_bad existing_factors",
            list(disable_update.get("existing_factors", [])),
            show_top=max(1, int(args.show_top)),
        )

    if bool(args.apply_remove_deprecated):
        remove_targets = set(deprecated_names)
        # If disabling bad factors in same run, also drop them from store to avoid stale model columns.
        if bool(args.apply_disable_bad):
            remove_targets.update(bad_factor_names)
        cleanup = cleanup_factor_store_columns(
            factor_dir=Path(result.get("factor_dir", "")),
            start=start,
            end=end,
            columns_to_remove=sorted(remove_targets),
        )
        actions["remove_deprecated"] = cleanup
        print(
            "applied remove_deprecated:",
            f"files_modified={cleanup.get('files_modified')}",
            f"removed_columns_total={cleanup.get('removed_columns_total')}",
        )
        removed_by_col = dict(cleanup.get("removed_by_column", {}))
        if removed_by_col:
            print("remove_deprecated removed_by_column:")
            ranked = sorted(removed_by_col.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            top_n = max(1, int(args.show_top))
            for name, cnt in ranked[:top_n]:
                print(f"- {name}: files={cnt}")
            if len(ranked) > top_n:
                print(f"... and {len(ranked) - top_n} more")
        else:
            print("remove_deprecated removed_by_column: []")

    if actions:
        result = run_factor_quality_scan(
            factor_cfg=factor_cfg,
            paths_cfg=paths_cfg,
            start=start,
            end=end,
            coverage_threshold=float(args.coverage_threshold),
            non_null_threshold=float(args.non_null_threshold),
            constant_day_threshold=float(args.constant_day_threshold),
            min_days_for_coverage_check=max(1, int(args.min_days_for_coverage_check)),
        )
        result["actions"] = actions
        print("post-action rescan:")
        _print_summary(result, show_top=max(1, int(args.show_top)))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {out_path}")

    # Non-zero when issues remain after optional actions.
    has_issues = bool(result.get("deprecated_factors")) or bool(result.get("bad_factors"))
    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
