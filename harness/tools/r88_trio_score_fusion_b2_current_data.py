"""Current-data B2 baseline for the frozen R88 P6/P3/Regsim score trio.

Historical shadow returns are deliberately not used as the execution baseline.
This tool keeps the three already-produced score trees fixed, then runs P6,
P3, Regsim, and their one equal-weight neutral-rank fusion through the same
current generic backtest contract. All comparison metrics are restricted to the
same 399 common score dates.

No model is trained or rescored. No live configuration, database, scheduler,
model state, factor store, or production result is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cbond_on.app.usecases import backtest_runtime  # noqa: E402
from cbond_on.core.config import load_config_file, resolve_config_file_path  # noqa: E402
from cbond_on.core.fees import load_fees_buy_sell_bps  # noqa: E402
from harness.tools import r88_trio_score_fusion_b1 as b1  # noqa: E402
from harness.tools import r88_trio_score_fusion_preflight as a0  # noqa: E402


DEFAULT_OUTPUT_ROOT = a0.DEFAULT_OUTPUT_ROOT
DEFAULT_A0_ROOT = DEFAULT_OUTPUT_ROOT / "a0_preflight_20260908_r1"
DEFAULT_RUN_NAME = "b2_current_data_equal_rank_20260908_r1"
MODEL_SCORE_ROOTS = {
    "P6_current_data": a0.R88_SCORE_ROOT / a0.MODELS["P6"],
    "P3_current_data": a0.R88_SCORE_ROOT / a0.MODELS["P3"],
    "Regsim_current_data": a0.REGSIM_SCORE_ROOT,
}


class B2Error(RuntimeError):
    """Raised when the current-data baseline cannot be aligned."""


def _json_default(value: object) -> object:
    return b1._json_default(value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _prepare_run_root(output_root: Path, run_name: str) -> Path:
    return b1._prepare_run_root(output_root, run_name)


def restrict_to_common_days(frame: pd.DataFrame, common_days: list[pd.Timestamp], *, label: str) -> pd.DataFrame:
    if "trade_date" not in frame.columns:
        raise B2Error(f"{label} lacks trade_date")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    if data["trade_date"].isna().any() or data["trade_date"].duplicated().any():
        raise B2Error(f"{label} has invalid or duplicate dates")
    result = data.set_index("trade_date").reindex(pd.DatetimeIndex(common_days)).reset_index().rename(columns={"index": "trade_date"})
    if result.isna().any().any():
        missing = result.loc[result.isna().any(axis=1), "trade_date"].dt.strftime("%Y-%m-%d").tolist()[:8]
        raise B2Error(f"{label} does not cover the locked common dates; first missing={missing}")
    return result


def _common_score_days(score_maps: Mapping[str, Mapping[pd.Timestamp, Path]]) -> list[pd.Timestamp]:
    days = sorted(set.intersection(*(set(score_maps[model]) for model in b1.MODELS)))
    if not days:
        raise B2Error("the frozen trio has no common score days")
    return days


def _score_root_for_label(label: str) -> Path:
    try:
        return MODEL_SCORE_ROOTS[label]
    except KeyError as exc:
        raise B2Error(f"unknown current-data baseline label: {label}") from exc


def _run_current_baseline(
    *,
    label: str,
    live: Mapping[str, Any],
    strategy: Mapping[str, Any],
    generic_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    common_days: list[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, Any], Path, int]:
    cfg = b1._generic_backtest_config(
        live=live,
        strategy=strategy,
        score_root=_score_root_for_label(label),
        output_root=generic_root,
        start=start,
        end=end,
        batch_id=f"r88_trio_b2_{label}",
    )
    result = backtest_runtime.run(start=start.date(), end=end.date(), cfg=cfg)
    raw = b1._load_generic_daily(result.out_dir)
    return restrict_to_common_days(raw, common_days, label=label), cfg, result.out_dir, len(raw)


def _write_results(run_root: Path, *, status: Mapping[str, Any], summary: pd.DataFrame, paired: pd.DataFrame, coverage: pd.DataFrame) -> None:
    def render_table(frame: pd.DataFrame) -> str:
        """Render a portable report table without making ``tabulate`` mandatory."""
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            return "```csv\n" + frame.to_csv(index=False) + "```"

    lines = [
        "# R88 P6 + Full88 P3 + Regsim B2 current-data unified baseline",
        "",
        "## Contract",
        "",
        "- Historical shadow-return history is not used as an execution baseline.",
        "- P6, P3, Regsim, and fusion use the same current generic runtime, current DataHub raw/pool inputs, o_0005, Top20, fees, benchmark, and strict cycle contract.",
        "- P6/P3/Regsim score trees are frozen inputs and are not retrained or rescored.",
        "- Fusion is only the fixed equal-weight own-universe percentile-rank blend with missing-model neutral rank 0.5.",
        "- All comparison rows use the same 399 common score days.",
        "",
        "## Status",
        "",
        "```json",
        json.dumps(status, ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        "## Aligned metrics",
        "",
        render_table(summary),
        "",
        "## Paired raw-return delta versus current Regsim",
        "",
        render_table(paired),
        "",
        "## Fusion neutral-fill coverage",
        "",
        render_table(coverage.describe().reset_index()),
        "",
    ]
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_b2(*, a0_root: Path = DEFAULT_A0_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT, run_name: str = DEFAULT_RUN_NAME) -> dict[str, Any]:
    paths_profile = b1._assert_paths_profile()
    manifest, verified_files = b1._verify_a0_manifest(a0_root)
    score_maps = b1._score_file_maps(manifest)
    common_days = _common_score_days(score_maps)
    start, end = common_days[0], common_days[-1]
    live = b1._read_json5(b1.LIVE_CONFIG)
    strategy = b1._read_json5(b1.STRATEGY_CONFIG)
    b1._validate_execution_contract(live, strategy)
    paths_cfg = load_config_file("paths")
    benchmark_path = resolve_config_file_path(b1.BENCHMARK_CONFIG_KEY)
    fees_path = resolve_config_file_path(b1.FEES_CONFIG_KEY)
    buy_bps, sell_bps, fees_source = load_fees_buy_sell_bps()
    run_root = _prepare_run_root(output_root, run_name)
    status_path = run_root / "run_status.json"
    run_manifest = {
        "schema": "r88_trio_score_fusion_b2_current_data_manifest/v1",
        "run_class": "research_only_current_data_unified_execution_baseline",
        "research_only": True,
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "model_training_called": False,
        "model_scoring_called": False,
        "a0_root": str(a0_root),
        "a0_input_manifest_sha256": a0.sha256(a0_root / "input_manifest.json"),
        "a0_reverified_input_file_count": verified_files,
        "candidate_membership": a0.MODELS,
        "baseline_definition": "all standalone scores and equal-rank fusion re-executed using current DataHub/raw/pool execution inputs; frozen historical shadow returns are not compared",
        "score_policy": {
            "normalization": "own-universe percentile rank",
            "target_universe": "current o_0005 universe for each score/trade day",
            "missing_score_contribution": b1.NEUTRAL_PERCENTILE_RANK,
            "equal_weights": {model: float(b1.EQUAL_WEIGHTS[index]) for index, model in enumerate(b1.MODELS)},
        },
        "common_period": {"start": str(start.date()), "end": str(end.date()), "days": len(common_days)},
        "current_runtime_inputs": {
            "paths_config": {"path": str(paths_profile), "sha256": a0.sha256(paths_profile)},
            "live_config": {"path": str(b1.LIVE_CONFIG), "sha256": a0.sha256(b1.LIVE_CONFIG)},
            "strategy_config": {"path": str(b1.STRATEGY_CONFIG), "sha256": a0.sha256(b1.STRATEGY_CONFIG)},
            "benchmark_config": {"path": str(benchmark_path), "sha256": a0.sha256(benchmark_path)},
            "fees_config": {"path": str(fees_path), "sha256": a0.sha256(fees_path)},
            "fees": {"buy_bps": buy_bps, "sell_bps": sell_bps, "source": fees_source},
        },
        "created_at_utc": datetime.now(timezone.utc),
    }
    _write_json(run_root / "run_manifest.json", run_manifest)
    status: dict[str, Any] = {
        "status": "running_current_data_standalones",
        "run_root": str(run_root),
        "research_only": True,
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "model_training_called": False,
        "model_scoring_called": False,
        "generic_backtest_called": True,
        "started_at_utc": datetime.now(timezone.utc),
    }
    _write_json(status_path, status)
    generic_root = run_root / "generic_backtests"
    current: dict[str, pd.DataFrame] = {}
    configs: dict[str, Any] = {}
    output_paths: dict[str, str] = {}
    raw_day_counts: dict[str, int] = {}
    for label in ("P6_current_data", "P3_current_data", "Regsim_current_data"):
        print(f"[B2] current-data generic baseline: {label}", flush=True)
        aligned, cfg, output, raw_count = _run_current_baseline(
            label=label,
            live=live,
            strategy=strategy,
            generic_root=generic_root,
            start=start,
            end=end,
            common_days=common_days,
        )
        current[label] = aligned
        configs[label] = cfg
        output_paths[label] = str(output)
        raw_day_counts[label] = int(raw_count)
    status["status"] = "writing_current_data_equal_rank_scores"
    _write_json(status_path, status)
    fused_root, coverage, fused_days = b1._build_fused_scores(
        score_maps=score_maps,
        live=live,
        paths_cfg=paths_cfg,
        run_root=run_root,
    )
    if fused_days != common_days:
        raise B2Error("fused score calendar drifted from the common current-data period")
    coverage.to_csv(run_root / "daily_fusion_coverage.csv", index=False)
    status["status"] = "running_current_data_equal_rank_fusion"
    _write_json(status_path, status)
    fusion_cfg = b1._generic_backtest_config(
        live=live,
        strategy=strategy,
        score_root=fused_root,
        output_root=generic_root,
        start=start,
        end=end,
        batch_id="r88_trio_b2_equal_rank_neutral_missing",
    )
    fusion_result = backtest_runtime.run(start=start.date(), end=end.date(), cfg=fusion_cfg)
    fusion_raw = b1._load_generic_daily(fusion_result.out_dir)
    current["equal_rank_neutral_missing"] = restrict_to_common_days(fusion_raw, common_days, label="equal_rank_neutral_missing")
    configs["equal_rank_neutral_missing"] = fusion_cfg
    output_paths["equal_rank_neutral_missing"] = str(fusion_result.out_dir)
    raw_day_counts["equal_rank_neutral_missing"] = int(len(fusion_raw))
    aligned = pd.concat([frame.assign(strategy=label) for label, frame in current.items()], ignore_index=True)
    aligned.to_csv(run_root / "current_data_aligned_returns.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for label, frame in current.items():
        summary_rows.extend(b1._summary_rows(label, frame, backtest_output=Path(output_paths[label])))
    summary = pd.DataFrame(summary_rows)
    regsim = current["Regsim_current_data"]
    paired_rows: list[dict[str, Any]] = []
    for label, frame in current.items():
        if label != "Regsim_current_data":
            paired_rows.extend(b1._paired_rows(frame, regsim, strategy=label))
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(run_root / "summary_metrics.csv", index=False)
    paired.to_csv(run_root / "paired_vs_regsim.csv", index=False)
    _write_json(run_root / "generic_backtest_configs.json", configs)
    status.update(
        {
            "status": "completed",
            "aligned_common_days": len(common_days),
            "raw_generic_day_counts": raw_day_counts,
            "generic_backtest_outputs": output_paths,
            "completed_at_utc": datetime.now(timezone.utc),
        }
    )
    _write_json(status_path, status)
    _write_results(run_root, status=status, summary=summary, paired=paired, coverage=coverage)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-root", type=Path, default=DEFAULT_A0_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_b2(a0_root=args.a0_root, output_root=args.output_root, run_name=str(args.run_name))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
