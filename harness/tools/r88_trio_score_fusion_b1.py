"""Research-only B1 equal-rank fusion for frozen R88 P6/P3/Regsim inputs.

The tool consumes the immutable A0 manifest, re-verifies every input hash,
replays raw Regsim scores through the generic backtest runtime, and proceeds to
one equal-weight rank fusion only after exact identity parity. It preserves the
existing o_0005 universe: each model ranks within its own valid daily score
universe, and an absent model score contributes the owner-approved neutral
percentile rank of 0.5 for that target-universe code.

All output stays under the dedicated research root. It never trains or
re-scores a model, writes a database, invokes live runtime, changes a live
configuration, or calls a scheduler.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import json5
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cbond_on.app.usecases import backtest_runtime  # noqa: E402
from cbond_on.core.config import load_config_file, resolve_config_file_path  # noqa: E402
from cbond_on.core.fees import load_fees_buy_sell_bps  # noqa: E402
from cbond_on.infra.universe.pool_filter import load_upstream_pool_config, resolve_pool_codes_for_trade_day  # noqa: E402
from harness.tools import r88_trio_score_fusion_preflight as a0  # noqa: E402
from harness.tools import validate_r88_single_model_phase1 as phase1  # noqa: E402


DEFAULT_OUTPUT_ROOT = a0.DEFAULT_OUTPUT_ROOT
DEFAULT_A0_ROOT = DEFAULT_OUTPUT_ROOT / "a0_preflight_20260908_r1"
DEFAULT_RUN_NAME = "b1_equal_rank_neutral_20260908_r1"
PATHS_CONFIG = REPO_ROOT / "cbond_on" / "config" / "data" / "paths_live50_20260805_config.json5"
LIVE_CONFIG = REPO_ROOT / "cbond_on" / "config" / "live" / "live_config.json5"
STRATEGY_CONFIG = REPO_ROOT / "cbond_on" / "config" / "strategies" / "strategy01" / "strategy01_config.json5"
BENCHMARK_CONFIG_KEY = "benchmark/benchmark"
FEES_CONFIG_KEY = "fees/fees"
TOP_K = 20
O005_ALLOWLIST_TABLE = "quant_factor_dev.researcher_xuvb.o_0005"
EPSILON = 1e-12
MODELS = ("P6", "P3", "Regsim")
EQUAL_WEIGHTS = np.full(len(MODELS), 1.0 / len(MODELS), dtype=float)
NEUTRAL_PERCENTILE_RANK = 0.5


class B1Error(RuntimeError):
    """Raised for a B1 contract or identity failure."""


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime, pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise B1Error(f"missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise B1Error(f"expected object JSON: {path}")
    return value


def _read_json5(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise B1Error(f"missing JSON5 input: {path}")
    value = json5.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise B1Error(f"expected object JSON5: {path}")
    return value


def _safe_git(args: Sequence[str]) -> str | None:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _assert_output_root(output_root: Path) -> Path:
    resolved = output_root.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise B1Error(f"B1 output must stay under {allowed}: {resolved}") from exc
    return resolved


def _prepare_run_root(output_root: Path, run_name: str) -> Path:
    output_root = _assert_output_root(output_root)
    if not run_name or Path(run_name).name != run_name:
        raise B1Error("run name must be exactly one non-empty directory leaf")
    run_root = _assert_output_root(output_root / run_name)
    if run_root.exists():
        raise B1Error(f"refusing to overwrite existing B1 run: {run_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=False, exist_ok=False)
    return run_root


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _assert_paths_profile() -> Path:
    configured = str(os.environ.get("CBOND_ON_PATHS_CONFIG", "")).strip()
    if not configured:
        raise B1Error("CBOND_ON_PATHS_CONFIG must explicitly select the live50 research input profile")
    actual = Path(configured).resolve()
    expected = PATHS_CONFIG.resolve()
    if actual != expected:
        raise B1Error(f"CBOND_ON_PATHS_CONFIG must be {expected}; got {actual}")
    return actual


def _verify_a0_manifest(a0_root: Path) -> tuple[dict[str, Any], int]:
    status = _read_json(a0_root / "run_status.json")
    contract = _read_json(a0_root / "study_contract.json")
    manifest = _read_json(a0_root / "input_manifest.json")
    if status.get("status") != "completed_a0_preflight":
        raise B1Error("A0 preflight is not completed")
    if contract.get("status") != "A0_PREFLIGHT_COMPLETE_STAGE_B_NOT_AUTHORIZED":
        raise B1Error("A0 study contract does not have the expected B1 gate state")
    if manifest.get("candidate_membership") != a0.MODELS:
        raise B1Error("A0 candidate membership no longer matches P6/P3/Regsim")
    checked = 0
    for records in manifest.get("score_files", {}).values():
        if not isinstance(records, list):
            raise B1Error("A0 score file manifest is malformed")
        for record in records:
            path = Path(str(record["path"]))
            if not path.is_file() or a0.sha256(path) != str(record["sha256"]):
                raise B1Error(f"A0 score input hash mismatch: {path}")
            checked += 1
    for record in manifest.get("return_files", {}).values():
        path = Path(str(record["path"]))
        if not path.is_file() or a0.sha256(path) != str(record["sha256"]):
            raise B1Error(f"A0 return input hash mismatch: {path}")
        checked += 1
    study = manifest.get("r88_study_integrity", {})
    integrity_path = Path(str(study.get("path", "")))
    if not integrity_path.is_file() or a0.sha256(integrity_path) != str(study.get("sha256", "")):
        raise B1Error(f"A0 R88 integrity hash mismatch: {integrity_path}")
    return manifest, checked + 1


def _score_file_maps(manifest: Mapping[str, Any]) -> dict[str, dict[pd.Timestamp, Path]]:
    result: dict[str, dict[pd.Timestamp, Path]] = {}
    for model in MODELS:
        records = manifest.get("score_files", {}).get(model)
        if not isinstance(records, list):
            raise B1Error(f"A0 manifest lacks score file records for {model}")
        mapping: dict[pd.Timestamp, Path] = {}
        for record in records:
            day = pd.Timestamp(record["score_day"]).normalize()
            if day in mapping:
                raise B1Error(f"duplicate manifest score day for {model}: {day.date()}")
            mapping[day] = Path(str(record["path"]))
        result[model] = mapping
    return result


def _load_score(path: Path, *, score_day: pd.Timestamp, model: str) -> pd.Series:
    try:
        frame = pd.read_csv(path, usecols=["trade_date", "code", "score"])
    except ValueError as exc:
        raise B1Error(f"{model} score file lacks required columns: {path}") from exc
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if frame.empty or frame["trade_date"].isna().any() or not frame["trade_date"].eq(score_day).all():
        raise B1Error(f"{model} score date mismatch: {path}")
    if frame["code"].eq("").any() or frame["code"].duplicated().any() or not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise B1Error(f"invalid {model} score payload: {path}")
    return frame.set_index("code")["score"].astype(float)


def _percentile_rank(series: pd.Series) -> pd.Series:
    if series.empty or series.index.has_duplicates or not np.isfinite(series.to_numpy(dtype=float)).all():
        raise B1Error("cannot percentile-rank an invalid score series")
    return series.rank(method="average", pct=True).astype(float)


def neutral_rank_fusion(
    score_by_model: Mapping[str, pd.Series],
    *,
    target_codes: Sequence[str],
    score_day: pd.Timestamp,
    weights: Sequence[float] = EQUAL_WEIGHTS,
    neutral_rank: float = NEUTRAL_PERCENTILE_RANK,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fuse own-universe percentile ranks over the unchanged target universe."""

    weight = np.asarray(weights, dtype=float)
    if weight.shape != (len(MODELS),) or not np.isfinite(weight).all() or np.any(weight < 0.0):
        raise B1Error("fusion weights must be a finite non-negative length-three simplex")
    if not math.isclose(float(weight.sum()), 1.0, abs_tol=1e-12):
        raise B1Error("fusion weights must sum to one")
    if not math.isfinite(float(neutral_rank)) or not 0.0 < float(neutral_rank) < 1.0:
        raise B1Error("neutral rank must lie strictly between zero and one")
    codes = sorted({str(code).strip() for code in target_codes if str(code).strip()})
    if len(codes) < TOP_K + 1:
        raise B1Error(f"target universe has fewer than {TOP_K + 1} codes on {score_day.date()}")
    columns: dict[str, pd.Series] = {}
    coverage: dict[str, int] = {}
    for model in MODELS:
        series = score_by_model.get(model)
        if not isinstance(series, pd.Series):
            raise B1Error(f"missing score series for {model}")
        ranks = _percentile_rank(series)
        aligned = ranks.reindex(codes).fillna(float(neutral_rank)).astype(float)
        columns[model] = aligned
        coverage[model] = int(ranks.index.isin(codes).sum())
    rank_frame = pd.DataFrame(columns, index=codes)
    blended = rank_frame.to_numpy(dtype=float) @ weight
    if not np.isfinite(blended).all():
        raise B1Error("neutral-rank fusion produced non-finite values")
    output = pd.DataFrame({"trade_date": score_day.strftime("%Y-%m-%d"), "code": codes, "score": blended})
    audit = {
        "score_day": str(score_day.date()),
        "target_universe_count": len(codes),
        **{f"{model.lower()}_raw_score_count": int(len(score_by_model[model])) for model in MODELS},
        **{f"{model.lower()}_target_coverage_count": coverage[model] for model in MODELS},
        **{f"{model.lower()}_neutral_fill_count": int(len(codes) - coverage[model]) for model in MODELS},
        "neutral_rank": float(neutral_rank),
        **{f"weight_{model}": float(weight[index]) for index, model in enumerate(MODELS)},
    }
    return output, audit


def _validate_execution_contract(live: Mapping[str, Any], strategy: Mapping[str, Any]) -> None:
    allowlist = live.get("allowlist")
    if not isinstance(allowlist, Mapping) or not bool(allowlist.get("enabled", True)):
        raise B1Error("current live config lacks an enabled allowlist")
    if str(allowlist.get("table", "")) != O005_ALLOWLIST_TABLE:
        raise B1Error("current live config is not the required o_0005 allowlist contract")
    if int(strategy.get("top_k", 0)) != TOP_K:
        raise B1Error("current strategy is not Top20")
    if not math.isclose(float(strategy.get("max_weight", float("nan"))), 0.05, abs_tol=0.0):
        raise B1Error("current strategy is not capped at 5 percent per name")
    if not math.isclose(float(strategy.get("turnover_ratio", float("nan"))), 1.0, abs_tol=0.0):
        raise B1Error("current strategy is not full-turnover")


def _generic_backtest_config(*, live: Mapping[str, Any], strategy: Mapping[str, Any], score_root: Path, output_root: Path, start: pd.Timestamp, end: pd.Timestamp, batch_id: str) -> dict[str, Any]:
    _validate_execution_contract(live, strategy)
    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "batch_id": batch_id,
        "score_source": {"score_root": str(score_root)},
        "strategy_id": "strategy01_topk_turnover",
        "strategy_config": dict(strategy),
        "buy_twap_col": str(live.get("output", {}).get("buy_twap_col", "twap_1442_1457")),
        "sell_twap_col": str(live.get("output", {}).get("sell_twap_col", "twap_0930_0939")),
        "allowlist": dict(live["allowlist"]),
        "execution_lag_trading_days": 0,
        "freeze_signal_universe": False,
        "output_root": str(output_root),
    }


def _load_generic_daily(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "daily_returns.csv"
    if not path.is_file():
        raise B1Error(f"generic backtest did not write daily returns: {path}")
    frame = pd.read_csv(path)
    required = {"trade_date", "day_return", "benchmark_return"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise B1Error(f"generic daily returns missing {missing}: {path}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    for column in ("day_return", "benchmark_return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "day_return", "benchmark_return"]).sort_values("trade_date")
    if frame["trade_date"].duplicated().any():
        raise B1Error(f"duplicate generic return date: {path}")
    return frame.reset_index(drop=True)


def _expected_regsim_returns(manifest: Mapping[str, Any], *, locked_score_days: Sequence[pd.Timestamp]) -> pd.DataFrame:
    record = manifest.get("return_files", {}).get("Regsim")
    if not isinstance(record, Mapping):
        raise B1Error("A0 manifest lacks Regsim return record")
    expected = a0.read_returns(Path(str(record["path"])), label="Regsim")
    expected = expected.loc[expected["trade_date"].isin(set(locked_score_days)), ["trade_date", "day_return"]].copy()
    return expected.rename(columns={"trade_date": "score_day", "day_return": "frozen_regsim_return"}).sort_values("score_day").reset_index(drop=True)


def _identity_comparison(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = actual[["trade_date", "day_return"]].rename(columns={"trade_date": "score_day", "day_return": "generic_regsim_return"})
    comparison = expected.merge(observed, on="score_day", how="outer", validate="one_to_one", indicator=True)
    both = comparison.loc[comparison["_merge"].eq("both")].copy()
    both["return_difference"] = both["generic_regsim_return"] - both["frozen_regsim_return"]
    maximum = float(both["return_difference"].abs().max()) if len(both) else float("inf")
    passed = bool(len(comparison) == len(expected) and comparison["_merge"].eq("both").all() and len(both) == len(expected) and math.isfinite(maximum) and maximum <= EPSILON)
    return comparison, {"status": "passed" if passed else "failed", "expected_days": int(len(expected)), "generic_days": int(len(actual)), "aligned_days": int(len(both)), "max_abs_return_difference": maximum}


def _scope(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "development_2025":
        return frame.loc[frame["trade_date"] <= pd.Timestamp("2025-12-31")].copy()
    if name == "reporting_2026":
        return frame.loc[frame["trade_date"] >= pd.Timestamp("2026-01-01")].copy()
    if name == "overall":
        return frame.copy()
    raise ValueError(name)


def _summary_rows(strategy: str, frame: pd.DataFrame, *, backtest_output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in ("development_2025", "reporting_2026", "overall"):
        subset = _scope(frame, scope)
        if len(subset) < 2:
            raise B1Error(f"{strategy} has too few {scope} returns")
        rows.append({"strategy": strategy, "scope": scope, "backtest_output": str(backtest_output), **phase1.calculate_metrics(subset)})
    return rows


def _paired_rows(candidate: pd.DataFrame, regsim: pd.DataFrame, *, strategy: str) -> list[dict[str, Any]]:
    return [
        {"strategy": strategy, "scope": scope, **phase1._raw_delta_hac(_scope(candidate, scope), _scope(regsim, scope))}
        for scope in ("development_2025", "reporting_2026", "overall")
    ]


def _build_fused_scores(*, score_maps: Mapping[str, Mapping[pd.Timestamp, Path]], live: Mapping[str, Any], paths_cfg: Mapping[str, Any], run_root: Path) -> tuple[Path, pd.DataFrame, list[pd.Timestamp]]:
    score_days = sorted(set.intersection(*(set(score_maps[model]) for model in MODELS)))
    if not score_days:
        raise B1Error("P6/P3/Regsim have no common score dates")
    allowlist = live.get("allowlist")
    if not isinstance(allowlist, Mapping):
        raise B1Error("live allowlist is missing")
    pool_cfg = load_upstream_pool_config(dict(allowlist))
    raw_data_root = str(paths_cfg["raw_data_root"])
    fused_root = run_root / "fused_score_inputs" / "equal_rank_neutral_missing"
    audit_rows: list[dict[str, Any]] = []
    for position, score_day in enumerate(score_days, start=1):
        pool_codes, pool_info = resolve_pool_codes_for_trade_day(raw_data_root=raw_data_root, trade_day=score_day.date(), pool_cfg=pool_cfg, enabled=True)
        if bool(pool_info.get("fallback_no_filter", False)):
            raise B1Error(f"required o_0005 pool unavailable during fusion input construction: day={score_day.date()} reason={pool_info.get('fallback_reason')}")
        score_by_model = {model: _load_score(score_maps[model][score_day], score_day=score_day, model=model) for model in MODELS}
        fused, audit = neutral_rank_fusion(score_by_model, target_codes=sorted(pool_codes), score_day=score_day)
        output = fused_root / score_day.strftime("%Y-%m") / f"{score_day:%Y-%m-%d}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        fused.to_csv(output, index=False)
        audit.update({"score_path": str(output), "score_sha256": a0.sha256(output), "allowlist_reference_day": str(pool_info.get("allowlist_day_used") or pool_info.get("pool_day_used") or ""), "allowlist_codes_count": int(pool_info.get("allowlist_codes_count") or pool_info.get("pool_codes_count") or 0)})
        audit_rows.append(audit)
        if position == 1 or position == len(score_days) or position % 50 == 0:
            print(f"[B1] fused scores {position}/{len(score_days)} day={score_day:%Y-%m-%d}", flush=True)
    return fused_root, pd.DataFrame(audit_rows), score_days


def _write_results(run_root: Path, *, status: Mapping[str, Any], summary: pd.DataFrame, paired: pd.DataFrame, coverage: pd.DataFrame) -> None:
    lines = [
        "# R88 P6 + Full88 P3 + Regsim B1 equal-rank fusion", "",
        "## Contract", "",
        "- Research-only; no DB, live runtime, scheduler, model training, or model scoring call.",
        "- Existing o_0005 universe, masks, Top20, strict cycle, current fee, and benchmark runtime contracts are retained.",
        "- P6/P3/Regsim each use own-universe percentile ranks; an unavailable model score contributes neutral rank 0.5 for that existing o_0005 code.",
        "- This run contains only Regsim identity and equal-weight fusion; no parameter search, selector, threshold, BaseGap, or dynamic weighting.", "",
        "## Status", "", "```json", json.dumps(status, ensure_ascii=False, indent=2, default=_json_default), "```", "",
        "## Summary metrics", "", summary.to_markdown(index=False) if not summary.empty else "No fusion metrics: identity gate did not pass.", "",
        "## Paired raw-return delta versus Regsim", "", paired.to_markdown(index=False) if not paired.empty else "No paired fusion result: identity gate did not pass.", "",
        "## Neutral-fill coverage", "", coverage.describe().to_markdown() if not coverage.empty else "No fused-score coverage table.", "",
    ]
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_b1(*, a0_root: Path = DEFAULT_A0_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT, run_name: str = DEFAULT_RUN_NAME) -> dict[str, Any]:
    paths_profile = _assert_paths_profile()
    manifest, verified_files = _verify_a0_manifest(a0_root)
    score_maps = _score_file_maps(manifest)
    live = _read_json5(LIVE_CONFIG)
    strategy = _read_json5(STRATEGY_CONFIG)
    _validate_execution_contract(live, strategy)
    paths_cfg = load_config_file("paths")
    # The live50 paths profile deliberately retains its normal default results
    # root for ordinary runtime compatibility. Every B1 generic configuration
    # below supplies an explicit, asserted research-only output_root instead.
    common_score_days = sorted(set.intersection(*(set(score_maps[model]) for model in MODELS)))
    if not common_score_days:
        raise B1Error("no common trio score dates")
    start, end = common_score_days[0], common_score_days[-1]
    expected_identity = _expected_regsim_returns(manifest, locked_score_days=common_score_days)
    run_root = _prepare_run_root(output_root, run_name)
    status_path = run_root / "run_status.json"
    benchmark_path = resolve_config_file_path(BENCHMARK_CONFIG_KEY)
    fees_path = resolve_config_file_path(FEES_CONFIG_KEY)
    buy_bps, sell_bps, fees_source = load_fees_buy_sell_bps()
    run_manifest = {
        "schema": "r88_trio_score_fusion_b1_manifest/v1", "run_class": "research_only_equal_rank_neutral_missing_score_fusion",
        "research_only": True, "database_writes": False, "live_runtime_called": False, "scheduler_called": False,
        "model_training_called": False, "model_scoring_called": False, "a0_root": str(a0_root),
        "a0_input_manifest_sha256": a0.sha256(a0_root / "input_manifest.json"), "a0_reverified_input_file_count": verified_files,
        "candidate_membership": a0.MODELS,
        "score_policy": {"normalization": "each model ranks within its own valid daily score universe", "target_universe": "existing current o_0005 universe for the score/trade day; never the score-file intersection", "missing_score_contribution": NEUTRAL_PERCENTILE_RANK, "weights": {model: float(EQUAL_WEIGHTS[index]) for index, model in enumerate(MODELS)}},
        "period": {"start": str(start.date()), "end": str(end.date()), "common_score_days": len(common_score_days)},
        "current_runtime_inputs": {"paths_config": {"path": str(paths_profile), "sha256": a0.sha256(paths_profile)}, "live_config": {"path": str(LIVE_CONFIG), "sha256": a0.sha256(LIVE_CONFIG)}, "strategy_config": {"path": str(STRATEGY_CONFIG), "sha256": a0.sha256(STRATEGY_CONFIG)}, "benchmark_config": {"path": str(benchmark_path), "sha256": a0.sha256(benchmark_path)}, "fees_config": {"path": str(fees_path), "sha256": a0.sha256(fees_path)}, "fees": {"buy_bps": buy_bps, "sell_bps": sell_bps, "source": fees_source}},
        "git": {"head": _safe_git(["rev-parse", "HEAD"]), "status_porcelain": _safe_git(["status", "--porcelain"])}, "created_at_utc": datetime.now(timezone.utc),
    }
    _write_json(run_root / "run_manifest.json", run_manifest)
    status: dict[str, Any] = {"status": "running_regsim_identity", "run_root": str(run_root), "research_only": True, "database_writes": False, "live_runtime_called": False, "scheduler_called": False, "model_training_called": False, "model_scoring_called": False, "generic_backtest_called": True, "started_at_utc": datetime.now(timezone.utc)}
    _write_json(status_path, status)
    configs: dict[str, Any] = {}
    generic_root = run_root / "generic_backtests"
    regsim_cfg = _generic_backtest_config(live=live, strategy=strategy, score_root=a0.REGSIM_SCORE_ROOT, output_root=generic_root, start=start, end=end, batch_id="r88_trio_b1_regsim_identity")
    configs["Regsim_identity"] = regsim_cfg
    identity_result = backtest_runtime.run(start=start.date(), end=end.date(), cfg=regsim_cfg)
    identity_daily = _load_generic_daily(identity_result.out_dir)
    identity_table, identity = _identity_comparison(expected_identity, identity_daily)
    identity["generic_backtest_output"] = str(identity_result.out_dir)
    identity_table.to_csv(run_root / "identity_regsim_parity.csv", index=False)
    status["identity_regsim_parity"] = identity
    if identity["status"] != "passed":
        status["status"] = "blocked_nonparity"
        status["completed_at_utc"] = datetime.now(timezone.utc)
        _write_json(status_path, status)
        for name in ("daily_fusion_coverage.csv", "daily_score_level_returns.csv", "summary_metrics.csv", "paired_vs_regsim.csv"):
            pd.DataFrame().to_csv(run_root / name, index=False)
        _write_json(run_root / "generic_backtest_configs.json", configs)
        _write_results(run_root, status=status, summary=pd.DataFrame(), paired=pd.DataFrame(), coverage=pd.DataFrame())
        return status
    status["status"] = "writing_equal_rank_neutral_scores"
    _write_json(status_path, status)
    fused_root, coverage, fused_days = _build_fused_scores(score_maps=score_maps, live=live, paths_cfg=paths_cfg, run_root=run_root)
    if fused_days != common_score_days:
        raise B1Error("fused score calendar drifted from the locked common score calendar")
    coverage.to_csv(run_root / "daily_fusion_coverage.csv", index=False)
    status["status"] = "running_equal_rank_neutral_fusion"
    _write_json(status_path, status)
    fusion_cfg = _generic_backtest_config(live=live, strategy=strategy, score_root=fused_root, output_root=generic_root, start=start, end=end, batch_id="r88_trio_b1_equal_rank_neutral_missing")
    configs["equal_rank_neutral_missing"] = fusion_cfg
    fusion_result = backtest_runtime.run(start=start.date(), end=end.date(), cfg=fusion_cfg)
    fusion_daily = _load_generic_daily(fusion_result.out_dir)
    if fusion_daily["trade_date"].tolist() != identity_daily["trade_date"].tolist():
        raise B1Error("fusion generic backtest date coverage does not equal identity baseline")
    daily = pd.concat([identity_daily.assign(strategy="Regsim_identity"), fusion_daily.assign(strategy="equal_rank_neutral_missing")], ignore_index=True)
    summary = pd.DataFrame([*_summary_rows("Regsim_identity", identity_daily, backtest_output=identity_result.out_dir), *_summary_rows("equal_rank_neutral_missing", fusion_daily, backtest_output=fusion_result.out_dir)])
    paired = pd.DataFrame(_paired_rows(fusion_daily, identity_daily, strategy="equal_rank_neutral_missing"))
    daily.to_csv(run_root / "daily_score_level_returns.csv", index=False)
    summary.to_csv(run_root / "summary_metrics.csv", index=False)
    paired.to_csv(run_root / "paired_vs_regsim.csv", index=False)
    _write_json(run_root / "generic_backtest_configs.json", configs)
    status.update({"status": "completed", "identity_regsim_parity": identity, "fused_score_days": int(len(fused_days)), "generic_backtest_outputs": {"Regsim_identity": str(identity_result.out_dir), "equal_rank_neutral_missing": str(fusion_result.out_dir)}, "completed_at_utc": datetime.now(timezone.utc)})
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
    print(json.dumps(run_b1(a0_root=args.a0_root, output_root=args.output_root, run_name=str(args.run_name)), ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
