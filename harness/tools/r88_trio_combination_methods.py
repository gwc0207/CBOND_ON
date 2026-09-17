"""Research-only full method comparison for the frozen R88 P6/P3/Regsim trio.

The tool has two deliberately separate contracts:

* ``score_fusion`` creates one daily fused cross-sectional score, then invokes
  the existing generic strict Top20 backtest.  This is the primary contract.
* ``sleeve`` allocates capital across three independently traded books.  It is
  a diagnostic allocation contract and is never described as a single Top20.

All primary methods use B2's current-data 399-day contract.  Underlying model
scores remain frozen: this script never trains or re-scores P6, P3, or Regsim,
and it never calls live runtime, a scheduler, or a database writer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.special import softmax
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cbond_on.app.usecases import backtest_runtime  # noqa: E402
from cbond_on.core.config import load_config_file, resolve_config_file_path  # noqa: E402
from cbond_on.core.fees import load_fees_buy_sell_bps  # noqa: E402
from cbond_on.infra.benchmark.service import (  # noqa: E402
    compute_strict_cycle_detail_for_holdings,
    load_strict_market_day,
)
from cbond_on.infra.model.score_io import write_scores_by_date  # noqa: E402
from cbond_on.infra.universe.pool_filter import load_upstream_pool_config, resolve_pool_codes_for_trade_day  # noqa: E402
from harness import model_switch_temporal_arbitration as temporal  # noqa: E402
from harness.tools import r88_trio_score_fusion_b1 as b1  # noqa: E402
from harness.tools import r88_trio_score_fusion_b2_current_data as b2  # noqa: E402
from harness.tools import r88_trio_score_fusion_preflight as a0  # noqa: E402
from harness.tools import validate_r88_single_model_phase1 as phase1  # noqa: E402
from harness.tools import validate_r88_single_model_phase2 as phase2  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\r88_trio_combination_methods_20260908_r1")
DEFAULT_A0_ROOT = a0.DEFAULT_OUTPUT_ROOT / "a0_preflight_20260908_r1"
DEFAULT_B2_ROOT = a0.DEFAULT_OUTPUT_ROOT / "b2_current_data_equal_rank_20260908_r1"
DEFAULT_RUN_NAME = "full_methods_20260908_r1"
MODELS = ("P6", "P3", "Regsim")
BASE_RETURN_LABELS = {
    "P6": "P6_current_data",
    "P3": "P3_current_data",
    "Regsim": "Regsim_current_data",
}
EQUAL_WEIGHTS = np.full(3, 1.0 / 3.0, dtype=float)
NEUTRAL_RANK = 0.5
LOOKBACK_DAYS = 120
MIN_TRAIN_DAYS = 60
STATIC_GRID_STEP = 0.25
ANNUALIZATION = 252.0
BOOTSTRAP_REPS = 2_000
MCS_BOOTSTRAP_REPS = 2_000
BOOTSTRAP_BLOCK_LENGTH = 10
RANDOM_SEED = 20_260_908


class CombinationError(RuntimeError):
    """Raised when an experiment contract cannot be satisfied."""


@dataclass(frozen=True)
class ScoreMethod:
    method_id: str
    family: str
    description: str
    builder: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime, pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CombinationError(f"missing JSON input: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CombinationError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise CombinationError(f"JSON object required: {path}")
    return value


def _assert_run_root(output_root: Path, run_name: str, *, resume: bool) -> Path:
    root = output_root.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    if root != allowed:
        raise CombinationError(f"output root must be exactly {allowed}, got {root}")
    if not run_name or Path(run_name).name != run_name:
        raise CombinationError("run name must be exactly one non-empty directory leaf")
    run_root = root / run_name
    if run_root.exists() and not resume:
        raise CombinationError(f"refusing to overwrite existing research run root: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def _markdown_table(frame: pd.DataFrame) -> str:
    """Keep reports portable on machines without the optional tabulate package."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + frame.to_csv(index=False) + "```"


def _scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    if scope == "development_2025":
        return work.loc[work["trade_date"] <= pd.Timestamp("2025-12-31")].copy()
    if scope == "reporting_2026":
        return work.loc[work["trade_date"] >= pd.Timestamp("2026-01-01")].copy()
    if scope == "overall":
        return work
    raise ValueError(f"unknown scope: {scope}")


def _load_b2_inputs(b2_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    status = _read_json(b2_root / "run_status.json")
    manifest = _read_json(b2_root / "run_manifest.json")
    if status.get("status") != "completed":
        raise CombinationError("B2 run is not completed")
    forbidden = ("database_writes", "live_runtime_called", "scheduler_called", "model_training_called", "model_scoring_called")
    if any(bool(status.get(field, False)) for field in forbidden):
        raise CombinationError("B2 status violates research-only contract")
    aligned_path = b2_root / "current_data_aligned_returns.csv"
    coverage_path = b2_root / "daily_fusion_coverage.csv"
    if not aligned_path.is_file() or not coverage_path.is_file():
        raise CombinationError("B2 aligned returns or coverage artifact is missing")
    aligned = pd.read_csv(aligned_path)
    coverage = pd.read_csv(coverage_path)
    aligned["trade_date"] = pd.to_datetime(aligned["trade_date"], errors="coerce").dt.normalize()
    coverage["score_day"] = pd.to_datetime(coverage["score_day"], errors="coerce").dt.normalize()
    if aligned["trade_date"].isna().any() or coverage["score_day"].isna().any():
        raise CombinationError("B2 contains invalid dates")
    expected_labels = set(BASE_RETURN_LABELS.values()) | {"equal_rank_neutral_missing"}
    observed = set(aligned["strategy"].astype(str))
    if observed != expected_labels:
        raise CombinationError(f"unexpected B2 strategy labels: {sorted(observed)}")
    date_sets = {label: tuple(group["trade_date"]) for label, group in aligned.groupby("strategy", sort=True)}
    if len(set(date_sets.values())) != 1:
        raise CombinationError("B2 strategies do not share an identical date calendar")
    dates = pd.DatetimeIndex(next(iter(date_sets.values())))
    if len(dates) != 399 or not dates.is_monotonic_increasing:
        raise CombinationError("B2 must provide exactly 399 ordered common dates")
    if tuple(coverage["score_day"]) != tuple(dates):
        raise CombinationError("B2 coverage calendar differs from aligned return calendar")
    return aligned, coverage, status, manifest


def _b2_base_returns(aligned: pd.DataFrame) -> pd.DataFrame:
    columns = ["trade_date", "benchmark_return"]
    pieces: list[pd.DataFrame] = []
    benchmark: pd.DataFrame | None = None
    for model, label in BASE_RETURN_LABELS.items():
        part = aligned.loc[aligned["strategy"].eq(label), ["trade_date", "day_return", "benchmark_return"]].copy()
        part = part.rename(columns={"day_return": model})
        if benchmark is None:
            benchmark = part[["trade_date", "benchmark_return"]].copy()
        elif not np.allclose(
            part["benchmark_return"].to_numpy(dtype=float),
            benchmark["benchmark_return"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-15,
        ):
            raise CombinationError("B2 base strategies do not share daily benchmark returns")
        pieces.append(part[["trade_date", model]])
    if benchmark is None:
        raise CombinationError("B2 base returns are absent")
    result = benchmark
    for part in pieces:
        result = result.merge(part, on="trade_date", how="inner", validate="one_to_one")
    if list(result.columns) != columns + list(MODELS):
        raise CombinationError("B2 base return layout is invalid")
    if len(result) != 399 or result.isna().any().any():
        raise CombinationError("B2 base returns do not form a finite 399-day panel")
    return result


def _current_config_hashes() -> dict[str, dict[str, str]]:
    paths_profile = b1._assert_paths_profile()
    benchmark_path = resolve_config_file_path(b1.BENCHMARK_CONFIG_KEY)
    fees_path = resolve_config_file_path(b1.FEES_CONFIG_KEY)
    paths = {
        "paths_config": paths_profile,
        "live_config": b1.LIVE_CONFIG,
        "strategy_config": b1.STRATEGY_CONFIG,
        "benchmark_config": benchmark_path,
        "fees_config": fees_path,
    }
    return {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()}


def _read_manifest_gate(path: Path, *, expected_day: pd.Timestamp, kind: str) -> dict[str, Any]:
    record = _read_json(path)
    if str(record.get("trade_day", "")) != expected_day.strftime("%Y-%m-%d"):
        raise CombinationError(f"{kind} manifest trade day mismatch: {path}")
    return record


def _raw_day_path(raw_root: Path, dataset: str, day: pd.Timestamp) -> Path:
    path = raw_root / dataset / day.strftime("%Y-%m") / f"{day:%Y%m%d}.parquet"
    if not path.is_file():
        raise CombinationError(f"missing current DataHub input partition: {path}")
    return path


def _parquet_schema_hash(path: Path) -> str:
    try:
        import pyarrow.parquet as pq

        schema = str(pq.ParquetFile(path).schema_arrow)
    except Exception as exc:  # pragma: no cover - environment/data corruption path
        raise CombinationError(f"cannot inspect parquet schema: {path}") from exc
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def write_as_run_input_audit(
    *,
    run_root: Path,
    b2_root: Path,
    aligned: pd.DataFrame,
    coverage: pd.DataFrame,
    b2_status: Mapping[str, Any],
    b2_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a no-copy audit of every actual DataHub input used by this study."""

    audit_root = run_root / "as_run_input_audit"
    if (audit_root / "audit_status.json").is_file():
        existing = _read_json(audit_root / "audit_status.json")
        if existing.get("status") == "completed":
            return existing
        raise CombinationError(f"input audit has incomplete prior state: {audit_root}")
    audit_root.mkdir(parents=True, exist_ok=False)
    source = aligned.loc[aligned["strategy"].eq("equal_rank_neutral_missing")].copy()
    source["buy_day"] = pd.to_datetime(source["buy_day"], errors="coerce").dt.normalize()
    source["sell_day"] = pd.to_datetime(source["sell_day"], errors="coerce").dt.normalize()
    pool_map = coverage.loc[:, ["score_day", "allowlist_reference_day"]].copy()
    pool_map["allowlist_reference_day"] = pd.to_datetime(pool_map["allowlist_reference_day"], errors="coerce").dt.normalize()
    source = source.merge(pool_map, left_on="trade_date", right_on="score_day", how="inner", validate="one_to_one")
    if len(source) != 399 or source[["buy_day", "sell_day", "allowlist_reference_day"]].isna().any().any():
        raise CombinationError("cannot derive the B2 source-date map")
    source = source.loc[:, ["trade_date", "buy_day", "sell_day", "allowlist_reference_day"]].sort_values("trade_date")
    source.to_csv(audit_root / "trade_day_source_map.csv", index=False)

    paths_cfg = load_config_file("paths")
    raw_root = Path(str(paths_cfg["raw_data_root"])).resolve()
    manifests_root = raw_root.parent / "manifests"
    market_days = sorted(set(source["buy_day"]) | set(source["sell_day"]))
    pool_days = sorted(set(source["allowlist_reference_day"]))
    all_days = sorted(set(market_days) | set(pool_days))
    if len(market_days) != 401 or len(pool_days) != 399 or len(all_days) != 403:
        raise CombinationError(
            f"unexpected B2 source-date cardinality market={len(market_days)} pool={len(pool_days)} all={len(all_days)}"
        )
    _write_json(
        audit_root / "audit_status.json",
        {
            "status": "running",
            "research_only": True,
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "started_at_utc": datetime.now(timezone.utc),
        },
    )

    manifest_rows: list[dict[str, Any]] = []
    for day in all_days:
        raw_manifest = _read_manifest_gate(manifests_root / "raw" / f"{day:%Y-%m-%d}.json", expected_day=day, kind="raw")
        clean_manifest = _read_manifest_gate(manifests_root / "clean" / f"{day:%Y-%m-%d}.json", expected_day=day, kind="clean")
        done = _read_manifest_gate(manifests_root / "publish" / f"{day:%Y-%m-%d}.done", expected_day=day, kind="publish")
        if raw_manifest.get("status") != "success" or clean_manifest.get("status") != "success":
            raise CombinationError(f"DataHub raw/clean manifest is not successful: {day:%Y-%m-%d}")
        if not bool(done.get("ready")) or bool(done.get("allow_partial_manifest", True)):
            raise CombinationError(f"DataHub publish gate is not strict-ready: {day:%Y-%m-%d}")
        required = set(done.get("require_datasets", []))
        if not {"raw", "clean"}.issubset(required):
            raise CombinationError(f"DataHub publish gate lacks raw/clean requirement: {day:%Y-%m-%d}")
        run_ids = {str(raw_manifest.get("run_id", "")), str(clean_manifest.get("run_id", "")), str(done.get("run_id", ""))}
        if len(run_ids) != 1 or "" in run_ids:
            raise CombinationError(f"DataHub run_id mismatch: {day:%Y-%m-%d}")
        for kind, path, payload in (
            ("raw_manifest", manifests_root / "raw" / f"{day:%Y-%m-%d}.json", raw_manifest),
            ("clean_manifest", manifests_root / "clean" / f"{day:%Y-%m-%d}.json", clean_manifest),
            ("publish_done", manifests_root / "publish" / f"{day:%Y-%m-%d}.done", done),
        ):
            stat = path.stat()
            manifest_rows.append(
                {
                    "trade_day": day,
                    "kind": kind,
                    "path": str(path),
                    "size_bytes": int(stat.st_size),
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "sha256": _sha256(path),
                    "run_id": str(payload.get("run_id", "")),
                }
            )
    manifest_inventory = pd.DataFrame(manifest_rows)
    manifest_inventory.to_csv(audit_root / "datahub_publish_evidence.csv", index=False)

    input_rows: list[dict[str, Any]] = []
    calendar = raw_root / "metadata__trading_calendar" / "all.parquet"
    if not calendar.is_file():
        raise CombinationError(f"missing trading calendar: {calendar}")
    input_specs: list[tuple[str, Path, str]] = [("trading_calendar", calendar, "")]
    for day in market_days:
        input_specs.append(("daily_twap", _raw_day_path(raw_root, "market_cbond__daily_twap", day), day.strftime("%Y-%m-%d")))
        input_specs.append(("daily_price", _raw_day_path(raw_root, "market_cbond__daily_price", day), day.strftime("%Y-%m-%d")))
    for day in pool_days:
        input_specs.append(("o_0005", _raw_day_path(raw_root, "quant_factor_dev__researcher_xuvb__o_0005", day), day.strftime("%Y-%m-%d")))
    for kind, path, day_text in input_specs:
        stat = path.stat()
        input_rows.append(
            {
                "kind": kind,
                "source_day": day_text,
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "schema_sha256": _parquet_schema_hash(path),
                "sha256": _sha256(path),
            }
        )
    input_inventory = pd.DataFrame(input_rows)
    if len(input_inventory) != 1 + 2 * 401 + 399:
        raise CombinationError("unexpected raw input inventory cardinality")
    input_inventory.to_csv(audit_root / "raw_input_file_inventory.csv", index=False)

    config_hashes = _current_config_hashes()
    expected_configs = b2_manifest.get("current_runtime_inputs", {})
    for name, record in config_hashes.items():
        expected = expected_configs.get(name, {}) if isinstance(expected_configs, Mapping) else {}
        if str(expected.get("sha256", "")) != record["sha256"]:
            raise CombinationError(f"current config hash drifted from B2: {name}")
    digest_rows = [
        *(f"input|{row.path}|{row.sha256}" for row in input_inventory.itertuples(index=False)),
        *(f"manifest|{row.path}|{row.sha256}" for row in manifest_inventory.itertuples(index=False)),
        *(f"config|{name}|{record['sha256']}" for name, record in config_hashes.items()),
        f"b2_manifest|{_sha256(b2_root / 'run_manifest.json')}",
        f"b2_aligned_returns|{_sha256(b2_root / 'current_data_aligned_returns.csv')}",
        f"b2_coverage|{_sha256(b2_root / 'daily_fusion_coverage.csv')}",
    ]
    digest = hashlib.sha256("\n".join(sorted(digest_rows)).encode("utf-8")).hexdigest()
    completed = {
        "status": "completed",
        "research_only": True,
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "source_dates": {"market": len(market_days), "pool": len(pool_days), "union": len(all_days)},
        "raw_input_file_count": int(len(input_inventory)),
        "publish_evidence_file_count": int(len(manifest_inventory)),
        "input_digest_sha256": digest,
        "config_hashes": config_hashes,
        "b2_status_completed_at_utc": b2_status.get("completed_at_utc"),
        "completed_at_utc": datetime.now(timezone.utc),
    }
    _write_json(audit_root / "audit_status.json", completed)
    return completed


def verify_as_run_input_audit(*, run_root: Path, checkpoint: str) -> dict[str, Any]:
    """Fail closed if an audited current DataHub input drifted during this run."""

    audit_root = run_root / "as_run_input_audit"
    audit = _read_json(audit_root / "audit_status.json")
    if audit.get("status") != "completed":
        raise CombinationError("as-run input audit is not completed")
    raw_inventory = pd.read_csv(audit_root / "raw_input_file_inventory.csv")
    manifest_inventory = pd.read_csv(audit_root / "datahub_publish_evidence.csv")
    mismatches: list[str] = []
    for row in pd.concat(
        [raw_inventory.loc[:, ["path", "sha256"]], manifest_inventory.loc[:, ["path", "sha256"]]],
        ignore_index=True,
    ).itertuples(index=False):
        path = Path(str(row.path))
        if not path.is_file() or _sha256(path) != str(row.sha256):
            mismatches.append(str(path))
            if len(mismatches) >= 8:
                break
    config_mismatch = []
    for name, record in _current_config_hashes().items():
        expected = audit.get("config_hashes", {}).get(name, {})
        if record["sha256"] != str(expected.get("sha256", "")):
            config_mismatch.append(name)
    result = {
        "checkpoint": checkpoint,
        "status": "passed" if not mismatches and not config_mismatch else "failed",
        "raw_and_manifest_files_checked": int(len(raw_inventory) + len(manifest_inventory)),
        "file_mismatch_examples": mismatches,
        "config_mismatches": config_mismatch,
        "verified_at_utc": datetime.now(timezone.utc),
    }
    _write_json(audit_root / f"reverification_{checkpoint}.json", result)
    if result["status"] != "passed":
        raise CombinationError(f"as-run input drift detected at {checkpoint}: {mismatches or config_mismatch}")
    return result


def _score_maps_from_a0(a0_root: Path) -> tuple[dict[str, dict[pd.Timestamp, Path]], list[pd.Timestamp]]:
    manifest, _ = b1._verify_a0_manifest(a0_root)
    maps = b1._score_file_maps(manifest)
    common = b2._common_score_days(maps)
    if len(common) != 399:
        raise CombinationError(f"expected 399 common frozen score days, got {len(common)}")
    return maps, common


def _top_codes(series: pd.Series, count: int = 20) -> set[str]:
    return set(series.sort_values(ascending=False, kind="mergesort").head(count).index.astype(str))


def build_rank_panel(
    *,
    run_root: Path,
    a0_root: Path,
    b2_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build once the target-universe neutral-rank panel shared by every method."""

    cache_root = run_root / "shared_inputs"
    panel_path = cache_root / "neutral_rank_panel.parquet"
    geometry_path = cache_root / "daily_score_geometry.csv"
    if panel_path.is_file() and geometry_path.is_file():
        panel = pd.read_parquet(panel_path)
        geometry = pd.read_csv(geometry_path)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
        geometry["trade_date"] = pd.to_datetime(geometry["trade_date"], errors="coerce").dt.normalize()
        if set(panel["trade_date"].unique()) == set(b2_dates) and len(geometry) == len(b2_dates):
            return panel, geometry
        raise CombinationError("existing rank panel does not match B2 calendar")
    cache_root.mkdir(parents=True, exist_ok=True)
    maps, common = _score_maps_from_a0(a0_root)
    if tuple(common) != tuple(b2_dates):
        raise CombinationError("A0 common score calendar drifted from B2 calendar")
    paths_cfg = load_config_file("paths")
    live = b1._read_json5(b1.LIVE_CONFIG)
    b1._validate_execution_contract(live, b1._read_json5(b1.STRATEGY_CONFIG))
    allowlist = live.get("allowlist")
    if not isinstance(allowlist, Mapping):
        raise CombinationError("live o_0005 allowlist is unavailable")
    pool_cfg = load_upstream_pool_config(dict(allowlist))
    raw_root = str(paths_cfg["raw_data_root"])
    panel_rows: list[pd.DataFrame] = []
    geometry_rows: list[dict[str, Any]] = []
    for index, day in enumerate(common, start=1):
        target_codes, pool_info = resolve_pool_codes_for_trade_day(
            raw_data_root=raw_root,
            trade_day=day.date(),
            pool_cfg=pool_cfg,
            enabled=True,
        )
        if bool(pool_info.get("fallback_no_filter", False)):
            raise CombinationError(f"o_0005 unavailable while building rank panel: {day:%Y-%m-%d}")
        codes = sorted(str(code).strip() for code in target_codes if str(code).strip())
        if len(codes) < 21:
            raise CombinationError(f"o_0005 target universe too small: {day:%Y-%m-%d}")
        source_scores = {model: b1._load_score(maps[model][day], score_day=day, model=model) for model in MODELS}
        rank_columns: dict[str, pd.Series] = {}
        raw_counts: dict[str, int] = {}
        coverages: dict[str, int] = {}
        for model in MODELS:
            ranks = b1._percentile_rank(source_scores[model])
            aligned = ranks.reindex(codes).fillna(NEUTRAL_RANK).astype(float)
            rank_columns[f"rank_{model}"] = aligned
            raw_counts[model] = int(len(ranks))
            coverages[model] = int(ranks.index.isin(codes).sum())
        day_panel = pd.DataFrame(rank_columns, index=pd.Index(codes, name="code")).reset_index()
        day_panel.insert(0, "trade_date", day)
        panel_rows.append(day_panel)
        ranked = day_panel.set_index("code")
        top_sets = {model: _top_codes(ranked[f"rank_{model}"]) for model in MODELS}
        row: dict[str, Any] = {
            "trade_date": day,
            "target_universe_count": len(codes),
            "allowlist_reference_day": str(pool_info.get("allowlist_day_used") or pool_info.get("pool_day_used") or ""),
        }
        for model in MODELS:
            row[f"{model.lower()}_raw_score_count"] = raw_counts[model]
            row[f"{model.lower()}_coverage_count"] = coverages[model]
            row[f"{model.lower()}_neutral_fill_count"] = len(codes) - coverages[model]
        for left, right in combinations(MODELS, 2):
            row[f"rank_corr_{left}_{right}"] = float(ranked[f"rank_{left}"].corr(ranked[f"rank_{right}"], method="spearman"))
            row[f"top20_overlap_{left}_{right}"] = int(len(top_sets[left] & top_sets[right]))
        geometry_rows.append(row)
        if index == 1 or index == len(common) or index % 50 == 0:
            print(f"[combo] rank panel {index}/{len(common)} day={day:%Y-%m-%d}", flush=True)
    panel = pd.concat(panel_rows, ignore_index=True)
    geometry = pd.DataFrame(geometry_rows)
    if panel.duplicated(["trade_date", "code"]).any() or panel.isna().any().any():
        raise CombinationError("rank panel has invalid values")
    panel.to_parquet(panel_path, index=False)
    geometry.to_csv(geometry_path, index=False)
    return panel, geometry


def build_cross_sectional_labels(
    *,
    run_root: Path,
    rank_panel: pd.DataFrame,
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    """Build strict current-execution per-code labels for score stacking.

    Labels use the same buy/sell price functions and current fee contract as
    generic backtest.  A label observed on day ``t`` is only made available to
    a stacker when producing a score for a strictly later day.
    """

    cache_root = run_root / "shared_inputs"
    label_path = cache_root / "strict_cross_sectional_cycle_labels.parquet"
    if label_path.is_file():
        labels = pd.read_parquet(label_path)
        labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce").dt.normalize()
        if labels["trade_date"].nunique() == 399 and not labels.isna().any().any():
            return labels
        raise CombinationError("existing cross-sectional label cache is invalid")
    paths_cfg = load_config_file("paths")
    live = b1._read_json5(b1.LIVE_CONFIG)
    b1._validate_execution_contract(live, b1._read_json5(b1.STRATEGY_CONFIG))
    buy_bps, sell_bps, _ = load_fees_buy_sell_bps()
    raw_root = str(paths_cfg["raw_data_root"])
    schedule = aligned.loc[
        aligned["strategy"].eq("equal_rank_neutral_missing"), ["trade_date", "sell_day"]
    ].copy()
    schedule["trade_date"] = pd.to_datetime(schedule["trade_date"], errors="coerce").dt.normalize()
    schedule["sell_day"] = pd.to_datetime(schedule["sell_day"], errors="coerce").dt.normalize()
    if len(schedule) != 399 or schedule.isna().any().any() or schedule["trade_date"].duplicated().any():
        raise CombinationError("B2 schedule is invalid for strict label construction")
    label_rows: list[pd.DataFrame] = []
    for index, row in enumerate(schedule.itertuples(index=False), start=1):
        day = pd.Timestamp(row.trade_date).normalize()
        sell_day = pd.Timestamp(row.sell_day).normalize()
        target = rank_panel.loc[rank_panel["trade_date"].eq(day), ["code"]].copy()
        market = load_strict_market_day(
            raw_data_root=raw_root,
            trade_day=day.date(),
            buy_bps=buy_bps,
            sell_bps=sell_bps,
        )
        for column in ("buy_price", "buy_close_price"):
            market[column] = pd.to_numeric(market[column], errors="coerce")
        market = market.loc[
            market["buy_price"].notna()
            & market["buy_close_price"].notna()
            & (market["buy_price"] > 0.0)
            & (market["buy_close_price"] > 0.0)
        ].copy()
        market["code"] = market["code"].astype(str).str.strip()
        market = target.merge(market, on="code", how="inner", validate="one_to_one")
        if market.empty:
            raise CombinationError(f"no tradable o_0005 codes for strict label: {day:%Y-%m-%d}")
        buy_columns = [
            "code",
            "buy_price",
            "buy_close_price",
            "buy_leg_ret_gross",
            "buy_leg_ret_net",
            "buy_cost_bps",
        ]
        holdings = market.loc[:, [column for column in buy_columns if column in market.columns]].copy()
        holdings["weight"] = 1.0
        holdings["score"] = 0.0
        holdings["rank"] = 0
        detail = compute_strict_cycle_detail_for_holdings(
            raw_data_root=raw_root,
            buy_day=day.date(),
            sell_day=sell_day.date(),
            buy_holdings=holdings,
            sell_bps=sell_bps,
        )
        detail["return_net"] = pd.to_numeric(detail["return_net"], errors="coerce")
        detail = detail.loc[detail["return_net"].notna(), ["code", "return_net"]].copy()
        if detail.empty:
            raise CombinationError(f"no strict labels after sell pricing: {day:%Y-%m-%d}")
        detail.insert(0, "trade_date", day)
        detail = detail.rename(columns={"return_net": "label_return_net"})
        label_rows.append(detail)
        if index == 1 or index == len(schedule) or index % 50 == 0:
            print(f"[combo] strict labels {index}/{len(schedule)} day={day:%Y-%m-%d}", flush=True)
    labels = pd.concat(label_rows, ignore_index=True)
    if labels.duplicated(["trade_date", "code"]).any() or labels.isna().any().any():
        raise CombinationError("strict label cache contains invalid records")
    labels.to_parquet(label_path, index=False)
    coverage = labels.groupby("trade_date", sort=True).size().rename("labeled_code_count").reset_index()
    coverage.to_csv(cache_root / "strict_cross_sectional_label_coverage.csv", index=False)
    return labels


def _simplex_grid(step: float = STATIC_GRID_STEP) -> list[np.ndarray]:
    if not 0.0 < step <= 1.0:
        raise ValueError("simplex step must lie in (0, 1]")
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, abs_tol=1e-12):
        raise ValueError("simplex step must divide one exactly")
    values: list[np.ndarray] = []
    for p6 in range(units + 1):
        for p3 in range(units - p6 + 1):
            regsim = units - p6 - p3
            values.append(np.array([p6, p3, regsim], dtype=float) / float(units))
    return values


def _weight_method_id(prefix: str, weights: Sequence[float]) -> str:
    encoded = "_".join(f"{model.lower()}{int(round(float(weight) * 100)):02d}" for model, weight in zip(MODELS, weights))
    return f"{prefix}_{encoded}"


def _weights_frame(dates: Sequence[pd.Timestamp], weights: np.ndarray, *, method_id: str) -> pd.DataFrame:
    matrix = np.asarray(weights, dtype=float)
    if matrix.shape != (len(dates), len(MODELS)):
        raise CombinationError(f"invalid weight matrix for {method_id}: {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < -1e-12).any() or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise CombinationError(f"invalid simplex weights for {method_id}")
    result = pd.DataFrame(matrix, columns=[f"weight_{model}" for model in MODELS])
    result.insert(0, "trade_date", pd.DatetimeIndex(dates))
    result.insert(1, "method_id", method_id)
    return result


def _scores_from_rank_weights(rank_panel: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    columns = [f"weight_{model}" for model in MODELS]
    work = rank_panel.merge(weights.loc[:, ["trade_date", *columns]], on="trade_date", how="inner", validate="many_to_one")
    if len(work) != len(rank_panel):
        raise CombinationError("rank panel and method weights do not share a full calendar")
    rank_matrix = work.loc[:, [f"rank_{model}" for model in MODELS]].to_numpy(dtype=float)
    weight_matrix = work.loc[:, columns].to_numpy(dtype=float)
    output = work.loc[:, ["trade_date", "code"]].copy()
    output["score"] = np.einsum("ij,ij->i", rank_matrix, weight_matrix)
    if output.isna().any().any() or not np.isfinite(output["score"].to_numpy(dtype=float)).all():
        raise CombinationError("rank-weight fusion produced invalid score values")
    return output


def _normal_score_equal(rank_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(sorted(rank_panel["trade_date"].unique()))
    ranks = rank_panel.loc[:, [f"rank_{model}" for model in MODELS]].to_numpy(dtype=float)
    normal_scores = norm.ppf(np.clip(ranks, 1e-4, 1.0 - 1e-4))
    output = rank_panel.loc[:, ["trade_date", "code"]].copy()
    output["score"] = normal_scores.mean(axis=1)
    weights = _weights_frame(dates, np.repeat(EQUAL_WEIGHTS[None, :], len(dates), axis=0), method_id="normal_score_equal")
    return output, weights


def _simplex_optimize(objective: Callable[[np.ndarray], float], *, start: np.ndarray | None = None) -> np.ndarray:
    x0 = EQUAL_WEIGHTS.copy() if start is None else np.asarray(start, dtype=float).copy()
    if x0.shape != (3,) or not np.isfinite(x0).all() or np.any(x0 < 0.0):
        x0 = EQUAL_WEIGHTS.copy()
    x0 = x0 / float(x0.sum())
    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 300, "ftol": 1e-12, "disp": False},
    )
    if not result.success or result.x.shape != (3,) or not np.isfinite(result.x).all() or np.any(result.x < -1e-10):
        return EQUAL_WEIGHTS.copy()
    output = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    if output.sum() <= 0.0:
        return EQUAL_WEIGHTS.copy()
    return output / float(output.sum())


def _ledoit_covariance(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 10 or not np.isfinite(values).all():
        return np.eye(3, dtype=float) * 1e-6
    covariance = LedoitWolf().fit(values).covariance_
    covariance = (covariance + covariance.T) / 2.0
    values_eigen, vectors = np.linalg.eigh(covariance)
    values_eigen = np.maximum(values_eigen, 1e-12)
    return (vectors * values_eigen) @ vectors.T


def _shrunk_softmax(utility: np.ndarray, *, shrinkage: float) -> np.ndarray:
    values = np.asarray(utility, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        return EQUAL_WEIGHTS.copy()
    scale = float(np.std(values, ddof=0))
    if scale <= 1e-12:
        posterior = EQUAL_WEIGHTS.copy()
    else:
        posterior = softmax(np.clip((values - float(np.mean(values))) / scale, -8.0, 8.0))
    result = float(shrinkage) * EQUAL_WEIGHTS + (1.0 - float(shrinkage)) * posterior
    return result / float(result.sum())


def _history_arrays(base_returns: pd.DataFrame, position: int) -> tuple[np.ndarray, np.ndarray]:
    start = max(0, position - LOOKBACK_DAYS)
    history = base_returns.iloc[start:position]
    raw = history.loc[:, list(MODELS)].to_numpy(dtype=float)
    benchmark = history["benchmark_return"].to_numpy(dtype=float)
    return raw, raw - benchmark[:, None]


def _drawdown_cdar(values: np.ndarray, *, level: float = 0.05) -> float:
    returns = np.asarray(values, dtype=float)
    if returns.size == 0 or not np.isfinite(returns).all():
        return float("nan")
    nav = np.cumprod(1.0 + returns)
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    count = max(1, int(math.ceil(len(drawdown) * level)))
    return float(np.mean(np.sort(drawdown)[:count]))


def _dynamic_weights_from_returns(base_returns: pd.DataFrame, *, method_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create daily simplex weights using strictly prior B2 expert returns."""

    dates = pd.DatetimeIndex(base_returns["trade_date"])
    matrix = np.empty((len(base_returns), 3), dtype=float)
    audit_rows: list[dict[str, Any]] = []
    for index, day in enumerate(dates):
        raw, relative = _history_arrays(base_returns, index)
        use_fallback = len(raw) < MIN_TRAIN_DAYS
        diagnostics: dict[str, Any] = {"trade_date": day, "method_id": method_id, "history_days": int(len(raw)), "warmup_fallback_equal": bool(use_fallback)}
        if use_fallback:
            weights = EQUAL_WEIGHTS.copy()
        elif method_id == "fixed_share_hedge120":
            weights = temporal.fixed_share_hedge_weights(relative, target_clip=0.02, eta=0.35, share=0.20)
        elif method_id == "bayesian_evidence120":
            mean = relative.mean(axis=0)
            standard_error = relative.std(axis=0, ddof=1) / math.sqrt(len(relative))
            z_score = np.divide(mean, standard_error, out=np.zeros(3, dtype=float), where=standard_error > 1e-12)
            weights = _shrunk_softmax(z_score, shrinkage=0.55)
            diagnostics.update({f"posterior_z_{model}": float(z_score[pos]) for pos, model in enumerate(MODELS)})
        elif method_id == "ewma_utility120":
            forecast, covariance, effective_n = temporal.ewma_utility_forecast(relative, half_life=20.0)
            scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
            utility = np.divide(forecast, scale, out=np.zeros(3, dtype=float), where=scale > 0.0)
            weights = _shrunk_softmax(utility, shrinkage=0.55)
            diagnostics.update({"ewma_half_life": 20.0, "ewma_effective_n": float(effective_n)})
        elif method_id == "mvo_score_weight120":
            covariance = _ledoit_covariance(relative)
            mean = relative.mean(axis=0)
            weights = _simplex_optimize(
                lambda value: float(-value @ mean + 12.0 * (value @ covariance @ value) + 0.20 * np.sum((value - EQUAL_WEIGHTS) ** 2))
            )
            diagnostics["covariance_trace"] = float(np.trace(covariance))
        elif method_id == "downside_score_weight120":
            downside = np.sqrt(np.mean(np.minimum(relative, 0.0) ** 2, axis=0))
            utility = np.divide(relative.mean(axis=0), downside, out=np.zeros(3, dtype=float), where=downside > 1e-12)
            weights = _shrunk_softmax(utility, shrinkage=0.50)
            diagnostics.update({f"downside_{model}": float(downside[pos]) for pos, model in enumerate(MODELS)})
        elif method_id == "cvar_score_weight120":
            cvars = []
            for column in range(3):
                values = relative[:, column]
                tail = np.sort(values)[: max(1, int(math.ceil(len(values) * 0.05)))]
                cvars.append(float(np.mean(tail)))
            utility = relative.mean(axis=0) + 0.75 * np.asarray(cvars, dtype=float)
            weights = _shrunk_softmax(utility, shrinkage=0.50)
            diagnostics.update({f"cvar_{model}": float(cvars[pos]) for pos, model in enumerate(MODELS)})
        elif method_id == "cdar_score_weight120":
            cdars = np.array([_drawdown_cdar(relative[:, column]) for column in range(3)], dtype=float)
            utility = relative.mean(axis=0) + 0.15 * cdars
            weights = _shrunk_softmax(utility, shrinkage=0.50)
            diagnostics.update({f"cdar_{model}": float(cdars[pos]) for pos, model in enumerate(MODELS)})
        elif method_id == "rolling_best_expert120_control":
            best = int(np.argmax(relative.mean(axis=0)))
            weights = np.zeros(3, dtype=float)
            weights[best] = 1.0
            diagnostics["selected_expert"] = MODELS[best]
        else:
            raise ValueError(f"unknown dynamic method: {method_id}")
        matrix[index] = weights
        diagnostics.update({f"weight_{model}": float(weights[pos]) for pos, model in enumerate(MODELS)})
        audit_rows.append(diagnostics)
    return _weights_frame(dates, matrix, method_id=method_id), pd.DataFrame(audit_rows)


def _cross_sectional_stacker(
    rank_panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    method_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit non-negative Ridge/NNLS score coefficients using only earlier labels."""

    features = rank_panel.merge(labels, on=["trade_date", "code"], how="inner", validate="one_to_one")
    expected_days = pd.DatetimeIndex(sorted(rank_panel["trade_date"].unique()))
    if features.empty or set(features["trade_date"].unique()) != set(expected_days):
        raise CombinationError("strict cross-sectional labels do not cover the rank panel calendar")
    dates = expected_days
    scores: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for index, day in enumerate(dates):
        current = rank_panel.loc[rank_panel["trade_date"].eq(day)].copy()
        prior_days = dates[max(0, index - LOOKBACK_DAYS) : index]
        train = features.loc[features["trade_date"].isin(prior_days)].copy()
        fallback = len(prior_days) < MIN_TRAIN_DAYS or len(train) < 1_000
        coefficient = EQUAL_WEIGHTS.copy()
        if not fallback:
            x = train.loc[:, [f"rank_{model}" for model in MODELS]].to_numpy(dtype=float) - NEUTRAL_RANK
            day_mean = train.groupby("trade_date")["label_return_net"].transform("mean")
            y = (pd.to_numeric(train["label_return_net"], errors="coerce") - day_mean).to_numpy(dtype=float) * 10_000.0
            valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if len(x) >= 1_000:
                if method_id == "ridge_positive_stack120":
                    model = Ridge(alpha=1_000.0, fit_intercept=False, positive=True)
                    model.fit(x, y)
                    coefficient = np.asarray(model.coef_, dtype=float)
                elif method_id == "nnls_stack120":
                    coefficient, _ = nnls(x, y)
                else:
                    raise ValueError(f"unknown stacker: {method_id}")
                if not np.isfinite(coefficient).all() or np.any(coefficient < 0.0) or float(coefficient.sum()) <= 1e-15:
                    coefficient = EQUAL_WEIGHTS.copy()
                    fallback = True
                else:
                    coefficient = coefficient / float(coefficient.sum())
            else:
                fallback = True
        current["score"] = current.loc[:, [f"rank_{model}" for model in MODELS]].to_numpy(dtype=float) @ coefficient
        scores.append(current.loc[:, ["trade_date", "code", "score"]])
        audit_rows.append(
            {
                "trade_date": day,
                "method_id": method_id,
                "history_days": int(len(prior_days)),
                "training_rows": int(len(train)),
                "warmup_fallback_equal": bool(fallback),
                **{f"weight_{model}": float(coefficient[pos]) for pos, model in enumerate(MODELS)},
            }
        )
    output = pd.concat(scores, ignore_index=True)
    weights = pd.DataFrame(audit_rows)
    return output, weights


def _lgbm_meta_weights(base_returns: pd.DataFrame, geometry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strongly regularised prior-day LGBM expert allocator.

    The model predicts each expert's same-cycle excess return from only that
    day's score geometry plus lagged expert-return statistics.  At date ``t``
    it is trained on rows whose realised return is strictly before ``t``.
    """

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - installation boundary
        raise CombinationError("LightGBM is unavailable for the confirmed meta-allocator branch") from exc
    dates = pd.DatetimeIndex(base_returns["trade_date"])
    geometry_lookup = geometry.set_index("trade_date").reindex(dates)
    if geometry_lookup.isna().all(axis=None):
        raise CombinationError("daily score geometry does not cover the B2 calendar")
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for position, day in enumerate(dates):
        _, relative = _history_arrays(base_returns, position)
        geometry_row = geometry_lookup.loc[day]
        numeric_geometry = {
            f"geometry_{key}": float(value)
            for key, value in geometry_row.items()
            if key not in {"allowlist_reference_day"} and np.isscalar(value) and pd.notna(value)
        }
        for model_position, model in enumerate(MODELS):
            values = relative[:, model_position]
            feature = {
                "trade_date": day,
                "model": model,
                "model_P6": float(model == "P6"),
                "model_P3": float(model == "P3"),
                "model_Regsim": float(model == "Regsim"),
                "prior_mean_5": float(np.mean(values[-5:])) if len(values) >= 5 else 0.0,
                "prior_mean_20": float(np.mean(values[-20:])) if len(values) >= 20 else 0.0,
                "prior_mean_60": float(np.mean(values[-60:])) if len(values) >= 60 else 0.0,
                "prior_vol_20": float(np.std(values[-20:], ddof=1)) if len(values) >= 20 else 0.0,
                "prior_vol_60": float(np.std(values[-60:], ddof=1)) if len(values) >= 60 else 0.0,
                **numeric_geometry,
                "target_excess_return": float(base_returns.iloc[position][model] - base_returns.iloc[position]["benchmark_return"]),
            }
            feature_rows.append(feature)
    features = pd.DataFrame(feature_rows)
    feature_columns = [column for column in features.columns if column not in {"trade_date", "model", "target_excess_return"}]
    matrix = np.empty((len(dates), 3), dtype=float)
    for position, day in enumerate(dates):
        prior_dates = dates[max(0, position - LOOKBACK_DAYS) : position]
        train = features.loc[features["trade_date"].isin(prior_dates)].copy()
        current = features.loc[features["trade_date"].eq(day)].sort_values("model", key=lambda values: values.map({model: index for index, model in enumerate(MODELS)}))
        fallback = len(prior_dates) < MIN_TRAIN_DAYS or len(train) < MIN_TRAIN_DAYS * len(MODELS)
        prediction = np.zeros(3, dtype=float)
        if not fallback:
            model = LGBMRegressor(
                objective="regression",
                n_estimators=30,
                learning_rate=0.05,
                num_leaves=3,
                max_depth=2,
                min_child_samples=30,
                reg_alpha=10.0,
                reg_lambda=100.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                n_jobs=1,
                device_type="cpu",
                verbosity=-1,
            )
            model.fit(train.loc[:, feature_columns], train["target_excess_return"])
            prediction = np.asarray(model.predict(current.loc[:, feature_columns]), dtype=float)
            if prediction.shape != (3,) or not np.isfinite(prediction).all():
                prediction = np.zeros(3, dtype=float)
                fallback = True
        weights = EQUAL_WEIGHTS.copy() if fallback else _shrunk_softmax(prediction, shrinkage=0.60)
        matrix[position] = weights
        rows.append(
            {
                "trade_date": day,
                "method_id": "lgbm_meta_allocator120",
                "history_days": int(len(prior_dates)),
                "training_rows": int(len(train)),
                "warmup_fallback_equal": bool(fallback),
                **{f"prediction_{model}": float(prediction[idx]) for idx, model in enumerate(MODELS)},
                **{f"weight_{model}": float(weights[idx]) for idx, model in enumerate(MODELS)},
            }
        )
    return _weights_frame(dates, matrix, method_id="lgbm_meta_allocator120"), pd.DataFrame(rows)


def _static_builder(method_id: str, weights: np.ndarray) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    def build(rank_panel: pd.DataFrame, _base: pd.DataFrame, _labels: pd.DataFrame, _geometry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.DatetimeIndex(sorted(rank_panel["trade_date"].unique()))
        schedule = _weights_frame(dates, np.repeat(weights[None, :], len(dates), axis=0), method_id=method_id)
        return _scores_from_rank_weights(rank_panel, schedule), schedule

    return build


def _dynamic_builder(method_id: str) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    def build(rank_panel: pd.DataFrame, base: pd.DataFrame, _labels: pd.DataFrame, _geometry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        schedule, audit = _dynamic_weights_from_returns(base, method_id=method_id)
        return _scores_from_rank_weights(rank_panel, schedule), audit

    return build


def _stacker_builder(method_id: str) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    def build(rank_panel: pd.DataFrame, _base: pd.DataFrame, labels: pd.DataFrame, _geometry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _cross_sectional_stacker(rank_panel, labels, method_id=method_id)

    return build


def _normal_builder(rank_panel: pd.DataFrame, _base: pd.DataFrame, _labels: pd.DataFrame, _geometry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _normal_score_equal(rank_panel)


def _lgbm_builder(rank_panel: pd.DataFrame, base: pd.DataFrame, _labels: pd.DataFrame, geometry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule, audit = _lgbm_meta_weights(base, geometry)
    return _scores_from_rank_weights(rank_panel, schedule), audit


def score_method_registry() -> list[ScoreMethod]:
    methods: list[ScoreMethod] = []
    for weights in _simplex_grid():
        method_id = _weight_method_id("static_rank", weights)
        methods.append(
            ScoreMethod(
                method_id=method_id,
                family="static_simplex",
                description="Pre-registered static own-universe percentile-rank simplex weight.",
                builder=_static_builder(method_id, weights),
            )
        )
    methods.append(
        ScoreMethod(
            method_id="normal_score_equal",
            family="normalization",
            description="Equal-weight inverse-normal transformed percentile-rank score fusion.",
            builder=_normal_builder,
        )
    )
    for method_id, description in (
        ("ridge_positive_stack120", "Strict-prior 120-day non-negative Ridge cross-sectional score stacking."),
        ("nnls_stack120", "Strict-prior 120-day non-negative least-squares cross-sectional score stacking."),
    ):
        methods.append(ScoreMethod(method_id=method_id, family="cross_sectional_stacking", description=description, builder=_stacker_builder(method_id)))
    for method_id, description in (
        ("fixed_share_hedge120", "Strict-prior fixed-share Hedge expert score weights."),
        ("bayesian_evidence120", "Strict-prior Bayesian-evidence expert score weights, shrunk to equal weight."),
        ("ewma_utility120", "Strict-prior EWMA utility expert score weights, shrunk to equal weight."),
        ("mvo_score_weight120", "Strict-prior shrinkage MVO expert score weights."),
        ("downside_score_weight120", "Strict-prior downside-risk expert score weights."),
        ("cvar_score_weight120", "Strict-prior CVaR-aware expert score weights."),
        ("cdar_score_weight120", "Strict-prior CDaR-aware expert score weights."),
        ("rolling_best_expert120_control", "No-threshold rolling best-expert control; high-overfit-risk diagnostic only."),
    ):
        methods.append(ScoreMethod(method_id=method_id, family="rolling_expert_weighting", description=description, builder=_dynamic_builder(method_id)))
    methods.append(
        ScoreMethod(
            method_id="lgbm_meta_allocator120",
            family="contextual_meta_allocator",
            description="Strongly regularised strict-prior rolling LGBM expert allocator using score geometry and lagged performance.",
            builder=_lgbm_builder,
        )
    )
    identifiers = [method.method_id for method in methods]
    if len(identifiers) != len(set(identifiers)):
        raise CombinationError("score-method registry has duplicate identifiers")
    return methods


def _validate_method_scores(scores: pd.DataFrame, *, dates: pd.DatetimeIndex, method_id: str) -> pd.DataFrame:
    required = {"trade_date", "code", "score"}
    if not required.issubset(scores.columns):
        raise CombinationError(f"{method_id} score input lacks {sorted(required)}")
    output = scores.loc[:, ["trade_date", "code", "score"]].copy()
    output["trade_date"] = pd.to_datetime(output["trade_date"], errors="coerce").dt.normalize()
    output["code"] = output["code"].astype(str).str.strip()
    output["score"] = pd.to_numeric(output["score"], errors="coerce")
    if output.isna().any().any() or output["code"].eq("").any() or output.duplicated(["trade_date", "code"]).any():
        raise CombinationError(f"{method_id} score input has invalid cells or duplicates")
    actual_dates = pd.DatetimeIndex(sorted(output["trade_date"].unique()))
    if tuple(actual_dates) != tuple(dates):
        raise CombinationError(f"{method_id} score input calendar does not equal B2 common dates")
    per_day = output.groupby("trade_date", sort=True).size()
    if (per_day < 21).any():
        raise CombinationError(f"{method_id} has fewer than 21 score codes on at least one day")
    return output.sort_values(["trade_date", "code"], kind="mergesort").reset_index(drop=True)


def _validate_generic_aligned(frame: pd.DataFrame, *, dates: pd.DatetimeIndex, method_id: str) -> pd.DataFrame:
    result = b2.restrict_to_common_days(frame, list(dates), label=method_id)
    if len(result) != len(dates) or tuple(result["trade_date"]) != tuple(dates):
        raise CombinationError(f"{method_id} generic daily return calendar drifted")
    for column in ("count", "intended_count"):
        if column not in result.columns or not result[column].eq(20).all():
            raise CombinationError(f"{method_id} did not maintain Top20 on every common date")
    for column, target in (("total_weight", 1.0), ("cash_weight", 0.0), ("fill_rate", 1.0)):
        if column not in result.columns or not np.allclose(result[column].to_numpy(dtype=float), target, rtol=0.0, atol=1e-12):
            raise CombinationError(f"{method_id} did not maintain required {column}")
    return result


def _method_summary_rows(method_id: str, family: str, frame: pd.DataFrame, *, output_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in ("development_2025", "reporting_2026", "overall"):
        subset = _scope(frame, scope)
        if len(subset) < 2:
            raise CombinationError(f"{method_id} has insufficient {scope} observations")
        rows.append({"method_id": method_id, "family": family, "scope": scope, "backtest_output": output_path, **phase1.calculate_metrics(subset)})
    return rows


def _b2_controls(aligned: pd.DataFrame, *, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    mapping = {
        "b2_p6_standalone": "P6_current_data",
        "b2_p3_standalone": "P3_current_data",
        "b2_regsim_standalone": "Regsim_current_data",
        "b2_equal_rank_baseline": "equal_rank_neutral_missing",
    }
    pieces: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for method_id, source_label in mapping.items():
        frame = aligned.loc[aligned["strategy"].eq(source_label)].copy()
        frame = _validate_generic_aligned(frame, dates=dates, method_id=method_id)
        frame["method_id"] = method_id
        frame["family"] = "b2_control"
        pieces.append(frame)
        metadata.append(
            {
                "method_id": method_id,
                "family": "b2_control",
                "description": f"Reused B2 current-data aligned output: {source_label}.",
                "score_fusion_contract": bool(source_label == "equal_rank_neutral_missing"),
                "backtest_output": str(frame["strategy"].iloc[0]),
            }
        )
    return pd.concat(pieces, ignore_index=True), metadata


def run_score_fusion_family(
    *,
    run_root: Path,
    aligned: pd.DataFrame,
    rank_panel: pd.DataFrame,
    labels: pd.DataFrame,
    geometry: pd.DataFrame,
    status: dict[str, Any],
) -> pd.DataFrame:
    """Run every registered score-fusion method through generic strict Top20."""

    score_root = run_root / "score_fusion"
    input_root = score_root / "score_inputs"
    audit_root = score_root / "method_audits"
    generic_root = score_root / "generic_backtests"
    for directory in (input_root, audit_root, generic_root):
        directory.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(sorted(rank_panel["trade_date"].unique()))
    base_returns = _b2_base_returns(aligned)
    if tuple(base_returns["trade_date"]) != tuple(dates):
        raise CombinationError("rank-panel and B2 base-return calendars differ")
    live = b1._read_json5(b1.LIVE_CONFIG)
    strategy = b1._read_json5(b1.STRATEGY_CONFIG)
    b1._validate_execution_contract(live, strategy)
    methods = score_method_registry()
    registry_rows = [
        {
            "method_id": method.method_id,
            "family": method.family,
            "description": method.description,
            "contract": "single_top20_score_fusion",
            "lookback_days": LOOKBACK_DAYS if method.family != "static_simplex" and method.family != "normalization" else 0,
        }
        for method in methods
    ]
    control_returns, control_metadata = _b2_controls(aligned, dates=dates)
    registry_rows.extend(control_metadata)
    pd.DataFrame(registry_rows).sort_values("method_id").to_csv(score_root / "method_registry.csv", index=False)

    completed = status.setdefault("score_method_runs", {})
    all_returns: list[pd.DataFrame] = [control_returns]
    summary_rows: list[dict[str, Any]] = []
    for control_id, control_group in control_returns.groupby("method_id", sort=True):
        summary_rows.extend(_method_summary_rows(control_id, "b2_control", control_group, output_path="reused_b2_aligned_output"))
    for position, method in enumerate(methods, start=1):
        method_dir = input_root / method.method_id
        score_path = method_dir / "scores.csv"
        weights_path = audit_root / f"{method.method_id}_weights.csv"
        method_record = completed.get(method.method_id, {}) if isinstance(completed, Mapping) else {}
        existing_output = Path(str(method_record.get("backtest_output", ""))) if isinstance(method_record, Mapping) else Path()
        if method_record.get("status") == "completed" and existing_output.is_dir() and (existing_output / "daily_returns.csv").is_file():
            raw = b1._load_generic_daily(existing_output)
            result = _validate_generic_aligned(raw, dates=dates, method_id=method.method_id)
            output_path = existing_output
            print(f"[combo] reuse completed score method {position}/{len(methods)} {method.method_id}", flush=True)
        else:
            print(f"[combo] build score method {position}/{len(methods)} {method.method_id}", flush=True)
            scores, weights = method.builder(rank_panel, base_returns, labels, geometry)
            scores = _validate_method_scores(scores, dates=dates, method_id=method.method_id)
            if score_path.exists():
                persisted = pd.read_csv(score_path)
                persisted = _validate_method_scores(persisted, dates=dates, method_id=method.method_id)
                if len(persisted) != len(scores) or not np.allclose(persisted["score"].to_numpy(dtype=float), scores["score"].to_numpy(dtype=float), rtol=0.0, atol=1e-12):
                    raise CombinationError(f"refusing to overwrite a mismatched persisted score input: {score_path}")
            else:
                method_dir.mkdir(parents=True, exist_ok=True)
                write_scores_by_date(score_path, scores, overwrite=False, dedupe=False)
            if weights_path.exists():
                persisted_weights = pd.read_csv(weights_path)
                if len(persisted_weights) != len(weights):
                    raise CombinationError(f"refusing to overwrite a mismatched persisted weight audit: {weights_path}")
            else:
                weights.to_csv(weights_path, index=False)
            cfg = b1._generic_backtest_config(
                live=live,
                strategy=strategy,
                score_root=score_path,
                output_root=generic_root,
                start=dates[0],
                end=dates[-1],
                batch_id=f"r88_combo_{method.method_id}",
            )
            result_obj = backtest_runtime.run(start=dates[0].date(), end=dates[-1].date(), cfg=cfg)
            output_path = result_obj.out_dir
            raw = b1._load_generic_daily(output_path)
            result = _validate_generic_aligned(raw, dates=dates, method_id=method.method_id)
            completed[method.method_id] = {
                "status": "completed",
                "method_id": method.method_id,
                "family": method.family,
                "score_path": str(score_path),
                "score_sha256": _sha256(score_path),
                "weights_path": str(weights_path),
                "backtest_output": str(output_path),
                "raw_generic_days": int(len(raw)),
                "aligned_days": int(len(result)),
                "completed_at_utc": datetime.now(timezone.utc),
            }
            _write_json(run_root / "run_status.json", status)
        result = result.copy()
        result["method_id"] = method.method_id
        result["family"] = method.family
        all_returns.append(result)
        summary_rows.extend(_method_summary_rows(method.method_id, method.family, result, output_path=str(output_path)))
    combined = pd.concat(all_returns, ignore_index=True)
    expected_per_method = len(dates)
    sizes = combined.groupby("method_id", sort=True).size()
    if not sizes.eq(expected_per_method).all():
        raise CombinationError("score-method aligned result coverage is incomplete")
    combined.to_csv(score_root / "aligned_daily_returns.csv", index=False)
    pd.DataFrame(summary_rows).sort_values(["scope", "family", "method_id"]).to_csv(score_root / "summary_metrics.csv", index=False)
    status["score_fusion_status"] = "completed"
    status["score_method_count"] = int(len(methods))
    _write_json(run_root / "run_status.json", status)
    return combined


def _sleeve_weight_for_method(history: np.ndarray, relative: np.ndarray, *, method_id: str) -> np.ndarray:
    if method_id == "sleeve_equal_weight":
        return EQUAL_WEIGHTS.copy()
    covariance = _ledoit_covariance(history)
    mean = history.mean(axis=0)
    if method_id == "sleeve_gmv_ledoitwolf120":
        return _simplex_optimize(lambda value: float(value @ covariance @ value))
    if method_id == "sleeve_mvo_shrunk120":
        shrunk_mean = 0.50 * mean
        return _simplex_optimize(
            lambda value: float(-value @ shrunk_mean + 12.0 * (value @ covariance @ value) + 0.15 * np.sum((value - EQUAL_WEIGHTS) ** 2))
        )
    if method_id == "sleeve_erc120":
        def erc_objective(value: np.ndarray) -> float:
            portfolio_variance = float(value @ covariance @ value)
            if portfolio_variance <= 1e-18:
                return float("inf")
            contribution = value * (covariance @ value) / math.sqrt(portfolio_variance)
            return float(np.sum((contribution - float(np.mean(contribution))) ** 2))

        return _simplex_optimize(erc_objective)
    if method_id == "sleeve_cvar120":
        def cvar_objective(value: np.ndarray) -> float:
            portfolio = history @ value
            tail = np.sort(portfolio)[: max(1, int(math.ceil(len(portfolio) * 0.05)))]
            return float(-np.mean(tail))

        return _simplex_optimize(cvar_objective)
    if method_id == "sleeve_cdar120":
        return _simplex_optimize(lambda value: float(-_drawdown_cdar(history @ value)))
    if method_id == "sleeve_robust_mvo120":
        uncertainty = history.std(axis=0, ddof=1) / math.sqrt(len(history))
        robust_mean = mean - 1.96 * uncertainty
        return _simplex_optimize(
            lambda value: float(-value @ robust_mean + 15.0 * (value @ covariance @ value) + 0.25 * np.sum((value - EQUAL_WEIGHTS) ** 2))
        )
    if method_id == "sleeve_bayesian_shrinkage120":
        standard_error = relative.std(axis=0, ddof=1) / math.sqrt(len(relative))
        z_score = np.divide(relative.mean(axis=0), standard_error, out=np.zeros(3, dtype=float), where=standard_error > 1e-12)
        return _shrunk_softmax(z_score, shrinkage=0.60)
    raise ValueError(f"unknown sleeve method: {method_id}")


def run_sleeve_family(*, run_root: Path, aligned: pd.DataFrame, status: dict[str, Any]) -> pd.DataFrame:
    """Calculate rolling three-sleeve allocation controls, separately from Top20."""

    sleeve_root = run_root / "sleeve_allocation"
    daily_path = sleeve_root / "aligned_daily_returns.csv"
    weights_path = sleeve_root / "daily_weights.csv"
    summary_path = sleeve_root / "summary_metrics.csv"
    if daily_path.is_file() and weights_path.is_file() and summary_path.is_file():
        existing = pd.read_csv(daily_path)
        existing["trade_date"] = pd.to_datetime(existing["trade_date"], errors="coerce").dt.normalize()
        if existing["trade_date"].nunique() and not existing.isna().any().any():
            return existing
        raise CombinationError("existing sleeve output is invalid")
    sleeve_root.mkdir(parents=True, exist_ok=True)
    base = _b2_base_returns(aligned)
    dates = pd.DatetimeIndex(base["trade_date"])
    method_ids = (
        "sleeve_equal_weight",
        "sleeve_gmv_ledoitwolf120",
        "sleeve_mvo_shrunk120",
        "sleeve_erc120",
        "sleeve_cvar120",
        "sleeve_cdar120",
        "sleeve_robust_mvo120",
        "sleeve_bayesian_shrinkage120",
    )
    all_returns: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    raw_matrix = base.loc[:, list(MODELS)].to_numpy(dtype=float)
    benchmark = base["benchmark_return"].to_numpy(dtype=float)
    for method_id in method_ids:
        matrix = np.empty((len(base), 3), dtype=float)
        audit_rows: list[dict[str, Any]] = []
        for index, day in enumerate(dates):
            raw, relative = _history_arrays(base, index)
            fallback = len(raw) < MIN_TRAIN_DAYS
            weights = EQUAL_WEIGHTS.copy() if fallback else _sleeve_weight_for_method(raw, relative, method_id=method_id)
            matrix[index] = weights
            audit_rows.append(
                {
                    "trade_date": day,
                    "method_id": method_id,
                    "history_days": int(len(raw)),
                    "warmup_fallback_equal": bool(fallback),
                    **{f"weight_{model}": float(weights[position]) for position, model in enumerate(MODELS)},
                }
            )
        weights_frame = pd.DataFrame(audit_rows)
        returns = pd.DataFrame(
            {
                "trade_date": dates,
                "method_id": method_id,
                "family": "three_sleeve_allocation",
                "day_return": np.einsum("ij,ij->i", raw_matrix, matrix),
                "benchmark_return": benchmark,
            }
        )
        all_weights.append(weights_frame)
        all_returns.append(returns)
        summary_rows.extend(_method_summary_rows(method_id, "three_sleeve_allocation", returns, output_path="daily return sleeve allocation"))
    output = pd.concat(all_returns, ignore_index=True)
    weights_output = pd.concat(all_weights, ignore_index=True)
    output.to_csv(daily_path, index=False)
    weights_output.to_csv(weights_path, index=False)
    pd.DataFrame(summary_rows).sort_values(["scope", "method_id"]).to_csv(summary_path, index=False)
    _write_json(
        sleeve_root / "run_manifest.json",
        {
            "contract": "three_independent_sleeves_not_single_top20",
            "base_return_source": str(DEFAULT_B2_ROOT / "current_data_aligned_returns.csv"),
            "methods": list(method_ids),
            "lookback_days": LOOKBACK_DAYS,
            "warmup_policy": "equal weight until 60 strictly-prior days are available",
            "research_only": True,
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
        },
    )
    status["sleeve_status"] = "completed"
    status["sleeve_method_count"] = int(len(method_ids))
    _write_json(run_root / "run_status.json", status)
    return output


def _wide_return_panel(data: pd.DataFrame, *, method_ids: Sequence[str]) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.Series]:
    subset = data.loc[data["method_id"].isin(method_ids), ["trade_date", "method_id", "day_return", "benchmark_return"]].copy()
    subset["trade_date"] = pd.to_datetime(subset["trade_date"], errors="coerce").dt.normalize()
    if subset.isna().any().any():
        raise CombinationError("method result panel has invalid values")
    benchmark = subset.pivot(index="trade_date", columns="method_id", values="benchmark_return")
    if benchmark.shape[1] != len(method_ids) or not benchmark.nunique(axis=1).eq(1).all():
        raise CombinationError("methods in a family do not share a daily benchmark")
    returns = subset.pivot(index="trade_date", columns="method_id", values="day_return").reindex(columns=list(method_ids))
    if returns.isna().any().any() or len(returns) < 30:
        raise CombinationError("method result panel lacks common finite daily returns")
    return pd.DatetimeIndex(returns.index), returns, benchmark.iloc[:, 0]


def _unique_return_representatives(
    data: pd.DataFrame,
    *,
    method_ids: Sequence[str],
    reference_id: str,
) -> tuple[list[str], pd.DataFrame]:
    """Collapse exactly identical daily-return sequences for family tests only.

    Performance ledgers preserve every pre-registered method.  Family-wide DSR,
    RC/SPA, MCS, and PBO must not inflate their nominal trial count with a
    method that deterministically produced the same complete return sequence.
    """

    _, returns, _ = _wide_return_panel(data, method_ids=method_ids)
    preferred = [
        reference_id,
        "b2_equal_rank_baseline",
        "b2_p6_standalone",
        "b2_p3_standalone",
        *method_ids,
    ]
    ordered = list(dict.fromkeys(preferred))
    representatives: list[str] = []
    rows: list[dict[str, Any]] = []
    for method_id in ordered:
        if method_id not in returns.columns:
            continue
        values = returns[method_id].to_numpy(dtype=float)
        representative = next(
            (
                existing
                for existing in representatives
                if np.array_equal(values, returns[existing].to_numpy(dtype=float), equal_nan=False)
            ),
            None,
        )
        if representative is None:
            representatives.append(method_id)
            representative = method_id
        rows.append(
            {
                "method_id": method_id,
                "formal_representative_method_id": representative,
                "is_formal_representative": bool(method_id == representative),
                "sequence_relation": "unique" if method_id == representative else "exact_duplicate_daily_return_sequence",
            }
        )
    if reference_id not in representatives:
        raise CombinationError("reference method was lost during return-sequence de-duplication")
    return representatives, pd.DataFrame(rows).sort_values(["formal_representative_method_id", "method_id"], kind="stable")


def _objective_scores(values: np.ndarray, benchmark: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Frozen CSCV objective: 60% min-max Sharpe + 40% min-max HAC alpha t."""

    sharpe = np.empty(values.shape[1], dtype=float)
    alpha_t = np.empty(values.shape[1], dtype=float)
    for index in range(values.shape[1]):
        series = values[:, index]
        std = float(np.std(series, ddof=1))
        sharpe[index] = float(np.mean(series) / std * math.sqrt(ANNUALIZATION)) if std > 0.0 else float("-inf")
        alpha_t[index] = float(phase2._segmented_ols_hac(series, benchmark, segment_ids=segments)["alpha_hac_t"])
    def minmax(array: np.ndarray) -> np.ndarray:
        finite = np.isfinite(array)
        if finite.sum() <= 1:
            return np.zeros_like(array)
        low = float(np.min(array[finite]))
        high = float(np.max(array[finite]))
        result = np.zeros_like(array)
        if high > low:
            result[finite] = (array[finite] - low) / (high - low)
        return result

    return 0.60 * minmax(sharpe) + 0.40 * minmax(alpha_t)


def _generic_pbo_cscv(returns: pd.DataFrame, benchmark: pd.Series, *, family: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Ten-block CSCV PBO for an arbitrary pre-registered method family."""

    if len(returns) != len(benchmark) or len(returns) < 100 or returns.shape[1] < 2:
        raise CombinationError("CSCV PBO needs at least 100 days and two methods")
    candidates = list(returns.columns)
    blocks = [np.asarray(block, dtype=int) for block in np.array_split(np.arange(len(returns)), 10)]
    values = returns.to_numpy(dtype=float)
    bench = benchmark.to_numpy(dtype=float)
    split_rows: list[dict[str, Any]] = []
    chosen_ranks: dict[str, list[float]] = {candidate: [] for candidate in candidates}
    selected_count: dict[str, int] = {candidate: 0 for candidate in candidates}
    for split_number, in_blocks in enumerate(combinations(range(10), 5), start=1):
        out_blocks = tuple(index for index in range(10) if index not in set(in_blocks))
        in_indices = np.concatenate([blocks[index] for index in in_blocks])
        out_indices = np.concatenate([blocks[index] for index in out_blocks])
        in_segments = np.concatenate([np.full(len(blocks[index]), position, dtype=int) for position, index in enumerate(in_blocks)])
        out_segments = np.concatenate([np.full(len(blocks[index]), position, dtype=int) for position, index in enumerate(out_blocks)])
        in_objective = _objective_scores(values[in_indices], bench[in_indices], in_segments)
        out_objective = _objective_scores(values[out_indices], bench[out_indices], out_segments)
        winner_index = int(np.argmax(in_objective))
        winner = candidates[winner_index]
        ranks = pd.Series(out_objective, index=candidates).rank(ascending=False, method="average")
        out_rank = float(ranks[winner])
        percentile = (len(candidates) + 1.0 - out_rank) / (len(candidates) + 1.0)
        selected_count[winner] += 1
        chosen_ranks[winner].append(out_rank)
        split_rows.append(
            {
                "family": family,
                "split_number": split_number,
                "in_blocks": ",".join(str(value) for value in in_blocks),
                "out_blocks": ",".join(str(value) for value in out_blocks),
                "selected_method": winner,
                "out_selected_rank": out_rank,
                "out_selected_percentile": percentile,
                "out_selected_below_median": bool(out_rank > (len(candidates) + 1.0) / 2.0),
            }
        )
    splits = pd.DataFrame(split_rows)
    candidate_rows = [
        {
            "family": family,
            "method_id": candidate,
            "cscv_selected_count": int(selected_count[candidate]),
            "cscv_selected_oos_rank_mean": float(np.mean(chosen_ranks[candidate])) if chosen_ranks[candidate] else float("nan"),
        }
        for candidate in candidates
    ]
    summary = {
        "family": family,
        "method": "CSCV PBO on frozen 0.6 Sharpe + 0.4 HAC alpha-t objective",
        "scope": "development_2025",
        "candidate_count": int(len(candidates)),
        "chronological_block_count": 10,
        "in_sample_block_count": 5,
        "directed_split_count": int(len(splits)),
        "pbo": float(np.mean(splits["out_selected_below_median"])),
        "selected_oos_median_rank": float(splits["out_selected_rank"].median()),
        "selection_objective": "0.6 minmax(annualized Sharpe) + 0.4 minmax(HAC alpha t)",
    }
    return summary, splits, pd.DataFrame(candidate_rows)


def _psr_dsr_table(returns: pd.DataFrame, *, family: str, scope: str) -> pd.DataFrame:
    daily_sharpes = np.asarray([phase2._daily_sharpe(returns[column].to_numpy(dtype=float)) for column in returns.columns], dtype=float)
    rows: list[dict[str, Any]] = []
    for column in returns.columns:
        result = phase2._dsr(
            returns[column].to_numpy(dtype=float),
            trial_daily_sharpes=daily_sharpes,
            trial_count=len(returns.columns),
        )
        rows.append({"family": family, "scope": scope, "method_id": column, **result})
    return pd.DataFrame(rows)


def _family_relative_tests_generic(
    deltas: pd.DataFrame,
    *,
    family: str,
    scope: str,
    reps: int,
    block_length: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """White-RC and Hansen-SPA-style test with a variable-size method family."""

    matrix = deltas.to_numpy(dtype=float)
    candidates = list(deltas.columns)
    if len(candidates) < 2 or len(matrix) < 30 or not np.isfinite(matrix).all():
        raise CombinationError("relative family test requires finite common returns and two methods")
    means = matrix.mean(axis=0)
    lrv = np.empty(len(candidates), dtype=float)
    lags = np.empty(len(candidates), dtype=int)
    for index in range(len(candidates)):
        lrv[index], lags[index] = phase2._hac_mean_long_run_variance(matrix[:, index])
    lrv_sd = np.sqrt(lrv)
    eligible = lrv_sd > 1e-15
    root_n = math.sqrt(len(matrix))
    rc_observed = float(root_n * np.max(means))
    spa_t = np.full(len(candidates), np.nan, dtype=float)
    spa_t[eligible] = root_n * means[eligible] / lrv_sd[eligible]
    spa_observed = float(np.nanmax(spa_t)) if np.any(eligible) else 0.0
    loglog_threshold = math.sqrt(max(0.0, 2.0 * math.log(math.log(len(matrix)))))
    anchors = {
        "lower": np.where(eligible & (spa_t >= 0.0), means, 0.0),
        "consistent": np.where(eligible & (spa_t >= -loglog_threshold), means, 0.0),
        "upper": means.copy(),
    }
    centered = matrix - means[None, :]
    rng = np.random.default_rng(seed)
    rc_exceed = 0
    spa_exceed = {key: 0 for key in anchors}
    cursor = 0
    while cursor < reps:
        batch = min(250, reps - cursor)
        indices = phase2._circular_block_index_matrix(len(matrix), block_length=block_length, rows=batch, rng=rng)
        rc_values = root_n * np.max(centered[indices].mean(axis=1), axis=1)
        rc_exceed += int(np.sum(rc_values >= rc_observed))
        raw_means = matrix[indices].mean(axis=1)
        for name, anchor in anchors.items():
            standardized = np.full((batch, len(candidates)), -np.inf, dtype=float)
            standardized[:, eligible] = root_n * (raw_means[:, eligible] - anchor[None, eligible]) / lrv_sd[None, eligible]
            values = np.max(standardized, axis=1) if np.any(eligible) else np.zeros(batch, dtype=float)
            spa_exceed[name] += int(np.sum(values >= spa_observed))
        cursor += batch
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        rows.append(
            {
                "family": family,
                "scope": scope,
                "method_id": candidate,
                **phase2._relative_hac_metrics(matrix[:, index]),
                "relative_hac_lag": int(lags[index]),
                "relative_spa_t": float(spa_t[index]),
                "relative_spa_eligible": bool(eligible[index]),
            }
        )
    best = int(np.argmax(means))
    summary = {
        "family": family,
        "scope": scope,
        "candidate_count": int(len(candidates)),
        "common_days": int(len(matrix)),
        "reference": "family-specific fixed reference",
        "white_reality_check_p": float((rc_exceed + 1) / (reps + 1)),
        "hansen_spa_p_lower": float((spa_exceed["lower"] + 1) / (reps + 1)),
        "hansen_spa_p_consistent": float((spa_exceed["consistent"] + 1) / (reps + 1)),
        "hansen_spa_p_upper": float((spa_exceed["upper"] + 1) / (reps + 1)),
        "best_mean_method": candidates[best],
        "best_mean_delta_bp_per_day": float(means[best] * 10_000.0),
        "bootstrap_reps": int(reps),
        "bootstrap_block_length": int(block_length),
    }
    return summary, pd.DataFrame(rows)


def _rolling_risk_rows(returns: pd.DataFrame, benchmark: pd.Series, *, family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method_id in returns.columns:
        frame = pd.DataFrame({"trade_date": returns.index, "day_return": returns[method_id].to_numpy(), "benchmark_return": benchmark.to_numpy()})
        for window in (60, 120):
            for end in range(window - 1, len(frame)):
                subset = frame.iloc[end - window + 1 : end + 1]
                rows.append(
                    {
                        "family": family,
                        "method_id": method_id,
                        "window_days": window,
                        "end_trade_date": subset["trade_date"].iloc[-1],
                        **phase1.calculate_metrics(subset),
                    }
                )
    return pd.DataFrame(rows)


def _paired_vs_reference(data: pd.DataFrame, *, method_ids: Sequence[str], reference_id: str, family: str) -> pd.DataFrame:
    reference = data.loc[data["method_id"].eq(reference_id)].copy()
    rows: list[dict[str, Any]] = []
    for method_id in method_ids:
        if method_id == reference_id:
            continue
        candidate = data.loc[data["method_id"].eq(method_id)].copy()
        for scope in ("development_2025", "reporting_2026", "overall"):
            rows.append(
                {
                    "family": family,
                    "method_id": method_id,
                    "reference_id": reference_id,
                    "scope": scope,
                    **phase1._raw_delta_hac(_scope(candidate, scope), _scope(reference, scope)),
                }
            )
    return pd.DataFrame(rows)


def _family_validation(
    *,
    run_root: Path,
    data: pd.DataFrame,
    method_ids: Sequence[str],
    reference_id: str,
    family: str,
    include_static_pbo: bool,
) -> dict[str, Any]:
    """Write aligned metrics and family-wise diagnostics for one contract family."""

    family_root = run_root / "validation" / family
    family_root.mkdir(parents=True, exist_ok=True)
    all_method_ids = list(dict.fromkeys(method_ids))
    dates, all_returns, benchmark = _wide_return_panel(data, method_ids=all_method_ids)
    if reference_id not in all_returns.columns:
        raise CombinationError(f"reference {reference_id} is absent from {family}")
    formal_method_ids, duplicate_audit = _unique_return_representatives(
        data,
        method_ids=all_method_ids,
        reference_id=reference_id,
    )
    returns = all_returns.loc[:, formal_method_ids]
    duplicate_audit.to_csv(family_root / "duplicate_return_sequences.csv", index=False)
    _write_json(
        family_root / "formal_method_family.json",
        {
            "all_ledger_method_count": int(len(all_method_ids)),
            "formal_unique_return_sequence_count": int(len(formal_method_ids)),
            "formal_method_ids": formal_method_ids,
            "reference_id": reference_id,
            "deduplication_rule": "exact equality across the complete aligned daily raw-return sequence",
        },
    )
    summary_rows: list[dict[str, Any]] = []
    for method_id in all_method_ids:
        frame = pd.DataFrame(
            {
                "trade_date": dates,
                "day_return": all_returns[method_id].to_numpy(dtype=float),
                "benchmark_return": benchmark.to_numpy(dtype=float),
            }
        )
        summary_rows.extend(_method_summary_rows(method_id, family, frame, output_path="aligned daily return ledger"))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(family_root / "summary_metrics.csv", index=False)
    paired = _paired_vs_reference(data, method_ids=all_method_ids, reference_id=reference_id, family=family)
    paired.to_csv(family_root / "paired_vs_reference.csv", index=False)
    risk = _rolling_risk_rows(all_returns, benchmark, family=family)
    risk.to_csv(family_root / "rolling_risk_metrics.csv", index=False)
    psr_rows: list[pd.DataFrame] = []
    for scope in ("development_2025", "reporting_2026", "overall"):
        mask = dates <= pd.Timestamp("2025-12-31") if scope == "development_2025" else dates >= pd.Timestamp("2026-01-01") if scope == "reporting_2026" else np.ones(len(dates), dtype=bool)
        psr_rows.append(_psr_dsr_table(returns.loc[mask], family=family, scope=scope))
    psr_dsr = pd.concat(psr_rows, ignore_index=True)
    psr_dsr.to_csv(family_root / "psr_dsr.csv", index=False)

    relative_rows: list[pd.DataFrame] = []
    family_summaries: list[dict[str, Any]] = []
    mcs_summaries: list[dict[str, Any]] = []
    mcs_memberships: list[pd.DataFrame] = []
    mcs_traces: list[pd.DataFrame] = []
    candidates = [method_id for method_id in formal_method_ids if method_id != reference_id]
    for scope in ("development_2025", "reporting_2026", "overall"):
        mask = dates <= pd.Timestamp("2025-12-31") if scope == "development_2025" else dates >= pd.Timestamp("2026-01-01") if scope == "reporting_2026" else np.ones(len(dates), dtype=bool)
        scoped_returns = returns.loc[mask]
        deltas = scoped_returns.loc[:, candidates].subtract(scoped_returns[reference_id], axis=0)
        relative_summary, relative_detail = _family_relative_tests_generic(
            deltas,
            family=family,
            scope=scope,
            reps=BOOTSTRAP_REPS,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            seed=RANDOM_SEED + len(family_summaries),
        )
        relative_summary["reference_id"] = reference_id
        family_summaries.append(relative_summary)
        relative_rows.append(relative_detail)
        mcs_summary, membership, trace = phase2._mcs_range(
            scoped_returns.loc[:, formal_method_ids],
            scope=scope,
            reps=MCS_BOOTSTRAP_REPS,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            alpha=0.10,
            seed=RANDOM_SEED + 1_000 + len(mcs_summaries),
        )
        mcs_summary["family"] = family
        mcs_summaries.append(mcs_summary)
        membership = membership.copy()
        trace = trace.copy()
        if "family" not in membership.columns:
            membership.insert(0, "family", family)
        else:
            membership["family"] = family
        if "scope" not in membership.columns:
            membership.insert(1, "scope", scope)
        if "family" not in trace.columns:
            trace.insert(0, "family", family)
        else:
            trace["family"] = family
        if "scope" not in trace.columns:
            trace.insert(1, "scope", scope)
        mcs_memberships.append(membership)
        mcs_traces.append(trace)
    pd.concat(relative_rows, ignore_index=True).to_csv(family_root / "white_rc_hansen_spa_detail.csv", index=False)
    _write_json(family_root / "white_rc_hansen_spa_summary.json", family_summaries)
    _write_json(family_root / "mcs_summary.json", mcs_summaries)
    pd.concat(mcs_memberships, ignore_index=True).to_csv(family_root / "mcs_membership.csv", index=False)
    pd.concat(mcs_traces, ignore_index=True).to_csv(family_root / "mcs_trace.csv", index=False)

    pbo_summary: dict[str, Any] | None = None
    if include_static_pbo:
        development_mask = dates <= pd.Timestamp("2025-12-31")
        pbo_summary, pbo_splits, pbo_candidates = _generic_pbo_cscv(
            returns.loc[development_mask, formal_method_ids],
            benchmark.loc[development_mask],
            family=family,
        )
        _write_json(family_root / "cscv_pbo_summary.json", pbo_summary)
        pbo_splits.to_csv(family_root / "cscv_pbo_splits.csv", index=False)
        pbo_candidates.to_csv(family_root / "cscv_pbo_candidates.csv", index=False)
    return {
        "family": family,
        "reference_id": reference_id,
        "ledger_method_count": int(len(all_method_ids)),
        "formal_unique_return_sequence_count": int(len(formal_method_ids)),
        "common_days": int(len(dates)),
        "pbo": pbo_summary,
        "white_rc_spa": family_summaries,
        "mcs": mcs_summaries,
    }


def _write_final_report(run_root: Path, *, status: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    score_summary_path = run_root / "validation" / "single_top20_score_fusion" / "summary_metrics.csv"
    sleeve_summary_path = run_root / "validation" / "three_sleeve_allocation" / "summary_metrics.csv"
    score = pd.read_csv(score_summary_path) if score_summary_path.is_file() else pd.DataFrame()
    sleeve = pd.read_csv(sleeve_summary_path) if sleeve_summary_path.is_file() else pd.DataFrame()
    report_lines = [
        "# R88 P6 + Full88 P3 + Regsim 全方法组合比较",
        "",
        "## 合同",
        "",
        "- 所有单一 Top20 方法均以冻结 P6/P3/Regsim score 生成新的融合 score，并走当前 DataHub 的同一 generic strict backtest。",
        "- `o_0005`、Top20、5% 单票、全换手、费用、TWAP、benchmark 与原 mask 均不变。",
        "- 2025 为开发/滚动训练区间；2026 仅为已观察 reporting，不用于选权重或超参数。",
        "- 三 sleeve 结果是独立资金配置对照，不是一份单一 Top20。",
        "",
        "## 运行状态",
        "",
        "```json",
        json.dumps(status, ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        "## 单一 Top20 score-fusion 汇总",
        "",
        _markdown_table(score) if not score.empty else "尚未完成。",
        "",
        "## 三 sleeve 对照汇总",
        "",
        _markdown_table(sleeve) if not sleeve.empty else "尚未完成。",
        "",
        "## 家族检验摘要",
        "",
        "```json",
        json.dumps(validation, ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        "## 解释边界",
        "",
        "- White RC、Hansen SPA、MCS、PBO、PSR/DSR 只校正本次预注册方法族，不构成上线授权。",
        "- 当前输入以 as-run DataHub 审计清单固定；没有不可变历史字节快照时，不应表述为历史冻结回放。",
    ]
    (run_root / "RESULTS.md").write_text("\n".join(report_lines), encoding="utf-8")


def _write_final_report_ascii(run_root: Path, *, status: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    """Write an ASCII report that remains readable across Windows code pages."""

    score_summary_path = run_root / "validation" / "single_top20_score_fusion" / "summary_metrics.csv"
    sleeve_summary_path = run_root / "validation" / "three_sleeve_allocation" / "summary_metrics.csv"
    score = pd.read_csv(score_summary_path) if score_summary_path.is_file() else pd.DataFrame()
    sleeve = pd.read_csv(sleeve_summary_path) if sleeve_summary_path.is_file() else pd.DataFrame()
    lines = [
        "# R88 P6 + Full88 P3 + Regsim combination-method comparison",
        "",
        "## Contract",
        "",
        "- Each single-Top20 method forms one fused score from frozen P6/P3/Regsim inputs and uses the current DataHub generic strict backtest.",
        "- The o_0005 universe, Top20, 5 percent name cap, full turnover, fees, TWAP, benchmark, and market mask are unchanged.",
        "- 2025 is development and rolling-training only. 2026 is observed reporting only and is not used to select a weight or hyperparameter.",
        "- Sleeve results allocate across three independent books and are not a single-Top20 strategy.",
        "",
        "## Run status",
        "",
        "```json",
        json.dumps(status, ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        "## Single-Top20 score-fusion metrics",
        "",
        _markdown_table(score) if not score.empty else "Not completed.",
        "",
        "## Three-sleeve allocation metrics",
        "",
        _markdown_table(sleeve) if not sleeve.empty else "Not completed.",
        "",
        "## Formal family-validation summary",
        "",
        "```json",
        json.dumps(validation, ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        "## Interpretation boundary",
        "",
        "- Exact duplicate daily-return sequences remain in performance ledgers but collapse to one representative for DSR, RC/SPA, MCS, and PBO family tests.",
        "- White RC, Hansen SPA, MCS, PBO, and PSR/DSR correct only this pre-registered method family. They do not authorize a live change.",
        "- Current inputs are fixed by an as-run DataHub audit, not by an immutable historical byte snapshot.",
    ]
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_validation(*, run_root: Path, status: dict[str, Any]) -> dict[str, Any]:
    score_path = run_root / "score_fusion" / "aligned_daily_returns.csv"
    sleeve_path = run_root / "sleeve_allocation" / "aligned_daily_returns.csv"
    registry_path = run_root / "score_fusion" / "method_registry.csv"
    if not score_path.is_file() or not sleeve_path.is_file() or not registry_path.is_file():
        raise CombinationError("score family, sleeve family, or registry output is missing before validation")
    score_data = pd.read_csv(score_path)
    sleeve_data = pd.read_csv(sleeve_path)
    registry = pd.read_csv(registry_path)
    score_data["trade_date"] = pd.to_datetime(score_data["trade_date"], errors="coerce").dt.normalize()
    sleeve_data["trade_date"] = pd.to_datetime(sleeve_data["trade_date"], errors="coerce").dt.normalize()
    static_ids = registry.loc[registry["family"].eq("static_simplex"), "method_id"].tolist()
    static_ids.extend(["b2_equal_rank_baseline", "b2_regsim_standalone"])
    score_ids = registry.loc[registry["contract"].eq("single_top20_score_fusion"), "method_id"].tolist()
    score_ids.extend(["b2_p6_standalone", "b2_p3_standalone", "b2_regsim_standalone", "b2_equal_rank_baseline"])
    score_ids = list(dict.fromkeys(score_ids))
    static_validation = _family_validation(
        run_root=run_root,
        data=score_data,
        method_ids=static_ids,
        reference_id="b2_regsim_standalone",
        family="static_score_fusion",
        include_static_pbo=True,
    )
    score_validation = _family_validation(
        run_root=run_root,
        data=score_data,
        method_ids=score_ids,
        reference_id="b2_regsim_standalone",
        family="single_top20_score_fusion",
        include_static_pbo=False,
    )
    sleeve_ids = sorted(sleeve_data["method_id"].astype(str).unique().tolist())
    sleeve_validation = _family_validation(
        run_root=run_root,
        data=sleeve_data,
        method_ids=sleeve_ids,
        reference_id="sleeve_equal_weight",
        family="three_sleeve_allocation",
        include_static_pbo=False,
    )
    validation = {
        "static_score_fusion": static_validation,
        "single_top20_score_fusion": score_validation,
        "three_sleeve_allocation": sleeve_validation,
    }
    _write_json(run_root / "validation" / "validation_status.json", validation)
    status["validation_status"] = "completed"
    status["completed_at_utc"] = datetime.now(timezone.utc)
    _write_json(run_root / "run_status.json", status)
    _write_final_report_ascii(run_root, status=status, validation=validation)
    return validation


def _initial_status(*, run_root: Path, b2_root: Path, a0_root: Path) -> dict[str, Any]:
    return {
        "schema": "r88_trio_combination_methods/v1",
        "status": "running",
        "research_only": True,
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "model_training_called": False,
        "model_scoring_called": False,
        "run_root": str(run_root),
        "a0_root": str(a0_root),
        "b2_root": str(b2_root),
        "contract": {
            "score_models": list(MODELS),
            "score_policy": "own-universe percentile rank; o_0005 target universe; absent score contributes neutral 0.5",
            "single_top20": {"top_k": 20, "max_weight": 0.05, "turnover_ratio": 1.0},
            "execution": {"buy_twap": "twap_1442_1457", "sell_twap": "twap_0930_0939", "execution_lag_days": 0},
            "development": "2025 only",
            "reporting": "2026 observed reporting only; never used for selection",
        },
        "started_at_utc": datetime.now(timezone.utc),
    }


def run_experiment(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str = DEFAULT_RUN_NAME,
    a0_root: Path = DEFAULT_A0_ROOT,
    b2_root: Path = DEFAULT_B2_ROOT,
    stage: str = "all",
    resume: bool = False,
) -> dict[str, Any]:
    run_root = _assert_run_root(output_root, run_name, resume=resume)
    status_path = run_root / "run_status.json"
    if status_path.is_file():
        if not resume:
            raise CombinationError("existing run status requires explicit --resume")
        status = _read_json(status_path)
        if str(status.get("b2_root", "")) != str(b2_root) or str(status.get("a0_root", "")) != str(a0_root):
            raise CombinationError("refusing to resume with different A0/B2 inputs")
    else:
        status = _initial_status(run_root=run_root, b2_root=b2_root, a0_root=a0_root)
        _write_json(status_path, status)
    aligned, coverage, b2_status, b2_manifest = _load_b2_inputs(b2_root)
    audit_status = write_as_run_input_audit(
        run_root=run_root,
        b2_root=b2_root,
        aligned=aligned,
        coverage=coverage,
        b2_status=b2_status,
        b2_manifest=b2_manifest,
    )
    status["as_run_input_audit"] = {
        "status": audit_status["status"],
        "input_digest_sha256": audit_status["input_digest_sha256"],
        "raw_input_file_count": audit_status["raw_input_file_count"],
    }
    rank_panel, geometry = build_rank_panel(
        run_root=run_root,
        a0_root=a0_root,
        b2_dates=pd.DatetimeIndex(sorted(aligned.loc[aligned["strategy"].eq("equal_rank_neutral_missing"), "trade_date"])),
    )
    labels = build_cross_sectional_labels(run_root=run_root, rank_panel=rank_panel, aligned=aligned)
    status["prepare_status"] = "completed"
    status["rank_panel_rows"] = int(len(rank_panel))
    status["cross_sectional_label_rows"] = int(len(labels))
    _write_json(status_path, status)
    if stage == "prepare":
        return status
    if stage in {"score", "all"}:
        status["as_run_input_audit"]["before_score"] = verify_as_run_input_audit(run_root=run_root, checkpoint="before_score")
        _write_json(status_path, status)
        run_score_fusion_family(
            run_root=run_root,
            aligned=aligned,
            rank_panel=rank_panel,
            labels=labels,
            geometry=geometry,
            status=status,
        )
    if stage in {"sleeve", "all"}:
        status["as_run_input_audit"]["before_sleeve"] = verify_as_run_input_audit(run_root=run_root, checkpoint="before_sleeve")
        _write_json(status_path, status)
        run_sleeve_family(run_root=run_root, aligned=aligned, status=status)
    if stage in {"validate", "all"}:
        status["as_run_input_audit"]["before_validation"] = verify_as_run_input_audit(run_root=run_root, checkpoint="before_validation")
        _write_json(status_path, status)
        run_validation(run_root=run_root, status=status)
    has_complete_outputs = (
        status.get("score_fusion_status") == "completed"
        and status.get("sleeve_status") == "completed"
        and status.get("validation_status") == "completed"
    )
    status["status"] = "completed" if has_complete_outputs else f"completed_{stage}"
    _write_json(status_path, status)
    return status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--a0-root", type=Path, default=DEFAULT_A0_ROOT)
    parser.add_argument("--b2-root", type=Path, default=DEFAULT_B2_ROOT)
    parser.add_argument("--stage", choices=("prepare", "score", "sleeve", "validate", "all"), default="all")
    parser.add_argument("--resume", action="store_true", help="Resume a previously created task-owned research root.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_experiment(
        output_root=args.output_root,
        run_name=str(args.run_name),
        a0_root=args.a0_root,
        b2_root=args.b2_root,
        stage=str(args.stage),
        resume=bool(args.resume),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
