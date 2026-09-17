"""Research-only CVaR score-fusion hyperparameter study for R88 P6/P3/Regsim.

This driver is deliberately isolated from live.  It consumes frozen scores and
current-data generic execution inputs, produces one single-Top20 score book per
configuration, and uses only strictly prior realized returns to form weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cbond_on.app.usecases import backtest_runtime  # noqa: E402
from cbond_on.infra.model.score_io import write_scores_by_date  # noqa: E402
from harness.tools import r88_trio_combination_methods as combo  # noqa: E402
from harness.tools import r88_trio_score_fusion_b1 as b1  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\r88_cvar_hyperopt_20260909_r1")
DEFAULT_RUN_NAME = "full_grid_20260909_r1"
DEFAULT_A0_ROOT = combo.DEFAULT_A0_ROOT
DEFAULT_B2_ROOT = combo.DEFAULT_B2_ROOT
DEFAULT_SOURCE_STUDY_ROOT = Path(
    r"D:\cbond_on\research_scratch\r88_trio_combination_methods_20260908_r1"
    r"\full_methods_20260908_r1"
)
MODELS = combo.MODELS
EQUAL_WEIGHTS = combo.EQUAL_WEIGHTS
MIN_HISTORY_DAYS = 60
BETA_LOOKBACK_DAYS = 60
RAW_LOOKBACKS = (60, 120, 180)
RAW_TAILS = (0.05, 0.10)
RAW_LAMBDAS = (0.0, 0.75, 1.50)
RAW_SHRINKAGES = (0.25, 0.50, 0.75)
RESIDUAL_LOOKBACKS = (120, 180)
RESIDUAL_TAILS = (0.05, 0.10)
RESIDUAL_LAMBDAS = (0.75, 1.50)
RESIDUAL_SHRINKAGES = (0.50, 0.75)


class CvarHyperoptError(RuntimeError):
    """Raised when the locked CVaR research contract cannot be satisfied."""


@dataclass(frozen=True)
class CvarSpec:
    family: str
    lookback_days: int
    tail_fraction: float
    lambda_cvar: float
    equal_shrinkage: float

    @property
    def method_id(self) -> str:
        tail = int(round(self.tail_fraction * 100.0))
        lam = f"{self.lambda_cvar:.2f}".replace(".", "p")
        shrink = int(round(self.equal_shrinkage * 100.0))
        return f"cvar_{self.family}_l{self.lookback_days:03d}_q{tail:02d}_lam{lam}_sh{shrink:02d}"


def _json_default(value: object) -> object:
    return combo._json_default(value)


def _sha256(path: Path) -> str:
    return combo._sha256(path)


def _write_json(path: Path, value: object) -> None:
    combo._write_json(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    return combo._read_json(path)


def _assert_run_root(output_root: Path, run_name: str, *, resume: bool) -> Path:
    resolved = output_root.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved != allowed:
        raise CvarHyperoptError(f"output root must be exactly {allowed}, got {resolved}")
    if not run_name or Path(run_name).name != run_name:
        raise CvarHyperoptError("run name must be exactly one non-empty directory leaf")
    run_root = resolved / run_name
    if run_root.exists() and not resume:
        raise CvarHyperoptError(f"refusing to overwrite existing research root: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def raw_specs() -> list[CvarSpec]:
    return [
        CvarSpec("raw", lookback, tail, lambda_cvar, shrinkage)
        for lookback, tail, lambda_cvar, shrinkage in product(
            RAW_LOOKBACKS,
            RAW_TAILS,
            RAW_LAMBDAS,
            RAW_SHRINKAGES,
        )
    ]


def residual_specs() -> list[CvarSpec]:
    return [
        CvarSpec("residual", lookback, tail, lambda_cvar, shrinkage)
        for lookback, tail, lambda_cvar, shrinkage in product(
            RESIDUAL_LOOKBACKS,
            RESIDUAL_TAILS,
            RESIDUAL_LAMBDAS,
            RESIDUAL_SHRINKAGES,
        )
    ]


def all_specs() -> list[CvarSpec]:
    specs = [*raw_specs(), *residual_specs()]
    if len(specs) != 70 or len({spec.method_id for spec in specs}) != 70:
        raise CvarHyperoptError("pre-registered grid cardinality is invalid")
    return specs


def _tail_mean(values: np.ndarray, tail_fraction: float) -> tuple[float, int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise CvarHyperoptError("CVaR tail input must be non-empty and finite")
    if not 0.0 < tail_fraction < 0.5:
        raise CvarHyperoptError("tail fraction must lie in (0, 0.5)")
    count = max(1, int(math.ceil(len(array) * tail_fraction)))
    return float(np.mean(np.sort(array)[:count])), count


def build_pit_beta_adjusted_path(run_root: Path, base_returns: pd.DataFrame) -> pd.DataFrame:
    """Build a genuinely point-in-time Beta-adjusted alpha-proxy path.

    For return on date s, fit alpha/beta using only the 60 complete cycles
    before s. Store r(s) - beta_hat(s) * benchmark(s), retaining the model's
    alpha level. A later score day t can consume only rows s < t.
    """

    shared = run_root / "shared_inputs"
    path = shared / "pit_beta_adjusted_returns.csv"
    if path.is_file():
        existing = pd.read_csv(path)
        existing["trade_date"] = pd.to_datetime(existing["trade_date"], errors="coerce").dt.normalize()
        required = {"trade_date", *[f"alpha_proxy_{model}" for model in MODELS], *[f"beta_fit_{model}" for model in MODELS]}
        if len(existing) == len(base_returns) and required.issubset(existing.columns):
            return existing
        raise CvarHyperoptError("existing PIT beta-adjusted path is invalid")
    shared.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(base_returns["trade_date"])
    raw_all = base_returns.loc[:, list(MODELS)].to_numpy(dtype=float)
    benchmark_all = base_returns["benchmark_return"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for position, day in enumerate(dates):
        record: dict[str, Any] = {
            "trade_date": day,
            "beta_lookback_days": BETA_LOOKBACK_DAYS,
            "beta_history_days": int(min(position, BETA_LOOKBACK_DAYS)),
            "prior_only_pass": bool(position >= BETA_LOOKBACK_DAYS),
        }
        if position < BETA_LOOKBACK_DAYS:
            for model in MODELS:
                record[f"alpha_fit_{model}"] = float("nan")
                record[f"beta_fit_{model}"] = float("nan")
                record[f"alpha_proxy_{model}"] = float("nan")
                record[f"innovation_{model}"] = float("nan")
            record["fit_status"] = "insufficient_prior_cycles"
        else:
            history = raw_all[position - BETA_LOOKBACK_DAYS : position]
            market = benchmark_all[position - BETA_LOOKBACK_DAYS : position]
            if not np.isfinite(history).all() or not np.isfinite(market).all() or float(np.var(market)) <= 1e-12:
                for model in MODELS:
                    record[f"alpha_fit_{model}"] = float("nan")
                    record[f"beta_fit_{model}"] = float("nan")
                    record[f"alpha_proxy_{model}"] = float("nan")
                    record[f"innovation_{model}"] = float("nan")
                record["fit_status"] = "invalid_prior_beta_fit"
            else:
                design = np.c_[np.ones(BETA_LOOKBACK_DAYS), market]
                coefficients = np.linalg.pinv(design.T @ design) @ (design.T @ history)
                today_return = raw_all[position]
                today_benchmark = benchmark_all[position]
                for index, model in enumerate(MODELS):
                    alpha_hat = float(coefficients[0, index])
                    beta_hat = float(coefficients[1, index])
                    alpha_proxy = float(today_return[index] - beta_hat * today_benchmark)
                    innovation = float(today_return[index] - alpha_hat - beta_hat * today_benchmark)
                    record[f"alpha_fit_{model}"] = alpha_hat
                    record[f"beta_fit_{model}"] = beta_hat
                    record[f"alpha_proxy_{model}"] = alpha_proxy
                    record[f"innovation_{model}"] = innovation
                record["fit_status"] = "valid"
        rows.append(record)
    output = pd.DataFrame(rows)
    output.to_csv(path, index=False)
    assertions = pd.DataFrame(
        {
            "trade_date": dates,
            "target_return_in_beta_fit": False,
            "future_return_in_beta_fit": False,
            "prior_only_pass": output["prior_only_pass"],
            "fit_status": output["fit_status"],
        }
    )
    assertions.to_csv(shared / "pit_beta_temporal_assertions.csv", index=False)
    return output


def _history_for_spec(base_returns: pd.DataFrame, position: int, lookback_days: int) -> tuple[np.ndarray, np.ndarray]:
    start = max(0, position - int(lookback_days))
    history = base_returns.iloc[start:position]
    raw = history.loc[:, list(MODELS)].to_numpy(dtype=float)
    benchmark = history["benchmark_return"].to_numpy(dtype=float)
    return raw, benchmark


def variant_weights(
    base_returns: pd.DataFrame,
    spec: CvarSpec,
    *,
    pit_beta_path: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate strictly prior daily simplex weights for one CVaR specification."""

    dates = pd.DatetimeIndex(base_returns["trade_date"])
    matrix = np.empty((len(dates), len(MODELS)), dtype=float)
    audit_rows: list[dict[str, Any]] = []
    for position, day in enumerate(dates):
        raw, benchmark = _history_for_spec(base_returns, position, spec.lookback_days)
        fallback = len(raw) < MIN_HISTORY_DAYS
        diagnostics: dict[str, Any] = {
            "trade_date": day,
            "method_id": spec.method_id,
            "family": spec.family,
            "lookback_days": spec.lookback_days,
            "tail_fraction": spec.tail_fraction,
            "lambda_cvar": spec.lambda_cvar,
            "equal_shrinkage": spec.equal_shrinkage,
            "history_days": int(len(raw)),
            "warmup_fallback_equal": bool(fallback),
        }
        alpha_vector = np.full(len(MODELS), np.nan, dtype=float)
        beta_vector = np.full(len(MODELS), np.nan, dtype=float)
        if fallback:
            weights = EQUAL_WEIGHTS.copy()
            cvars = np.full(len(MODELS), np.nan, dtype=float)
            means = np.full(len(MODELS), np.nan, dtype=float)
            tail_count = 0
        else:
            if spec.family == "raw":
                signal_history = raw - benchmark[:, None]
                mean_history = signal_history
            elif spec.family == "residual":
                if pit_beta_path is None:
                    raise CvarHyperoptError("residual CVaR requires the PIT beta-adjusted path")
                prior_path = pit_beta_path.iloc[:position].copy()
                proxy_columns = [f"alpha_proxy_{model}" for model in MODELS]
                innovation_columns = [f"innovation_{model}" for model in MODELS]
                prior_path = prior_path.loc[
                    prior_path["fit_status"].eq("valid"),
                    ["trade_date", *proxy_columns, *innovation_columns],
                ]
                prior_path = prior_path.dropna(how="any").tail(spec.lookback_days)
                if len(prior_path) < MIN_HISTORY_DAYS:
                    fallback = True
                    weights = EQUAL_WEIGHTS.copy()
                    cvars = np.full(len(MODELS), np.nan, dtype=float)
                    means = np.full(len(MODELS), np.nan, dtype=float)
                    tail_count = 0
                    diagnostics["warmup_fallback_equal"] = True
                    diagnostics["residual_history_days"] = int(len(prior_path))
                    matrix[position] = weights
                    diagnostics["tail_count"] = int(tail_count)
                    diagnostics.update({f"alpha_fit_{model}": float(alpha_vector[index]) for index, model in enumerate(MODELS)})
                    diagnostics.update({f"beta_fit_{model}": float(beta_vector[index]) for index, model in enumerate(MODELS)})
                    diagnostics.update({f"weight_{model}": float(weights[index]) for index, model in enumerate(MODELS)})
                    audit_rows.append(diagnostics)
                    continue
                # Preserve historical alpha level once through alpha_proxy,
                # but measure tail risk on the PIT OLS innovation so alpha is
                # not counted again inside CVaR.
                mean_history = prior_path.loc[:, proxy_columns].to_numpy(dtype=float)
                signal_history = prior_path.loc[:, innovation_columns].to_numpy(dtype=float)
                current_state = pit_beta_path.iloc[position]
                alpha_vector = np.asarray([current_state[f"alpha_fit_{model}"] for model in MODELS], dtype=float)
                beta_vector = np.asarray([current_state[f"beta_fit_{model}"] for model in MODELS], dtype=float)
                diagnostics["residual_history_days"] = int(len(prior_path))
                diagnostics["residual_history_start"] = str(pd.Timestamp(prior_path["trade_date"].iloc[0]).date())
                diagnostics["residual_history_end"] = str(pd.Timestamp(prior_path["trade_date"].iloc[-1]).date())
            else:
                raise CvarHyperoptError(f"unknown CVaR signal family: {spec.family}")
            means = mean_history.mean(axis=0)
            cvar_items = [_tail_mean(signal_history[:, column], spec.tail_fraction) for column in range(len(MODELS))]
            cvars = np.asarray([item[0] for item in cvar_items], dtype=float)
            tail_count = int(cvar_items[0][1])
            if any(item[1] != tail_count for item in cvar_items):
                raise CvarHyperoptError("CVaR tail count drifted across models")
            utility = means + float(spec.lambda_cvar) * cvars
            weights = combo._shrunk_softmax(utility, shrinkage=float(spec.equal_shrinkage))
            diagnostics.update({f"mean_{model}": float(means[index]) for index, model in enumerate(MODELS)})
            diagnostics.update({f"cvar_{model}": float(cvars[index]) for index, model in enumerate(MODELS)})
            diagnostics.update({f"utility_{model}": float(utility[index]) for index, model in enumerate(MODELS)})
        matrix[position] = weights
        diagnostics["tail_count"] = int(tail_count)
        diagnostics.update({f"alpha_fit_{model}": float(alpha_vector[index]) for index, model in enumerate(MODELS)})
        diagnostics.update({f"beta_fit_{model}": float(beta_vector[index]) for index, model in enumerate(MODELS)})
        diagnostics.update({f"weight_{model}": float(weights[index]) for index, model in enumerate(MODELS)})
        audit_rows.append(diagnostics)
    return combo._weights_frame(dates, matrix, method_id=spec.method_id), pd.DataFrame(audit_rows)


def _weight_digest(weights: pd.DataFrame) -> str:
    columns = [f"weight_{model}" for model in MODELS]
    values = np.ascontiguousarray(weights.loc[:, columns].to_numpy(dtype=np.float64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _default_raw_spec() -> CvarSpec:
    return CvarSpec("raw", 120, 0.05, 0.75, 0.50)


def _verify_default_parity(weights: pd.DataFrame, source_study_root: Path) -> None:
    audit_path = source_study_root / "score_fusion" / "method_audits" / "cvar_score_weight120_weights.csv"
    if not audit_path.is_file():
        raise CvarHyperoptError(f"fixed CVaR weight audit is missing: {audit_path}")
    expected = pd.read_csv(audit_path)
    expected["trade_date"] = pd.to_datetime(expected["trade_date"], errors="coerce").dt.normalize()
    observed = weights.copy()
    observed["trade_date"] = pd.to_datetime(observed["trade_date"], errors="coerce").dt.normalize()
    columns = [f"weight_{model}" for model in MODELS]
    merged = observed.merge(expected.loc[:, ["trade_date", *columns]], on="trade_date", how="inner", suffixes=("_observed", "_expected"), validate="one_to_one")
    if len(merged) != len(observed):
        raise CvarHyperoptError("fixed CVaR default parity has calendar drift")
    for column in columns:
        if not np.allclose(merged[f"{column}_observed"], merged[f"{column}_expected"], rtol=0.0, atol=1e-12):
            raise CvarHyperoptError(f"fixed CVaR default parity failed: {column}")


def _verify_score_only_a0(a0_root: Path) -> dict[str, Any]:
    """Verify only frozen score/integrity inputs consumed by this study.

    A0's historical return manifest is intentionally not consumed: this study
    uses B2 current-data daily returns to build CVaR weights. Keeping that
    boundary explicit prevents an unrelated mutable historical return artifact
    from invalidating the frozen score contract.
    """

    manifest = _read_json(a0_root / "input_manifest.json")
    if manifest.get("candidate_membership") != combo.a0.MODELS:
        raise CvarHyperoptError("A0 candidate membership differs from frozen R88 trio")
    checked = 0
    for model, records in manifest.get("score_files", {}).items():
        if model not in MODELS or not isinstance(records, list):
            raise CvarHyperoptError("A0 score manifest is malformed")
        for record in records:
            path = Path(str(record.get("path", "")))
            if not path.is_file() or _sha256(path) != str(record.get("sha256", "")):
                raise CvarHyperoptError(f"frozen score hash mismatch: {path}")
            checked += 1
    integrity = manifest.get("r88_study_integrity", {})
    integrity_path = Path(str(integrity.get("path", "")))
    if not integrity_path.is_file() or _sha256(integrity_path) != str(integrity.get("sha256", "")):
        raise CvarHyperoptError("R88 study integrity hash mismatch")
    checked += 1
    return {
        "status": "passed",
        "consumed_score_and_integrity_file_count": checked,
        "unconsumed_historical_return_records": sorted(str(key) for key in manifest.get("return_files", {}).keys()),
        "a0_manifest_sha256": _sha256(a0_root / "input_manifest.json"),
    }


def _load_source_rank_panel(source_study_root: Path, *, expected_dates: pd.DatetimeIndex) -> pd.DataFrame:
    path = source_study_root / "shared_inputs" / "neutral_rank_panel.parquet"
    if not path.is_file():
        raise CvarHyperoptError(f"source neutral rank panel is missing: {path}")
    panel = pd.read_parquet(path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    required = {"trade_date", "code", *[f"rank_{model}" for model in MODELS]}
    if not required.issubset(panel.columns) or panel.isna().any().any() or panel.duplicated(["trade_date", "code"]).any():
        raise CvarHyperoptError("source neutral rank panel is invalid")
    dates = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))
    if tuple(dates) != tuple(expected_dates):
        raise CvarHyperoptError("source rank panel date calendar differs from B2")
    if len(panel) != 157_239:
        raise CvarHyperoptError(f"unexpected source rank panel row count: {len(panel)}")
    return panel


def _compare_source_input_audits(run_root: Path, source_study_root: Path) -> dict[str, Any]:
    """Accept an irrelevant calendar byte drift only when schedule map is equal."""

    old_root = source_study_root / "as_run_input_audit"
    new_root = run_root / "as_run_input_audit"
    old_inventory = pd.read_csv(old_root / "raw_input_file_inventory.csv")
    new_inventory = pd.read_csv(new_root / "raw_input_file_inventory.csv")
    old_map = old_inventory.set_index("path")["sha256"].to_dict()
    new_map = new_inventory.set_index("path")["sha256"].to_dict()
    if set(old_map) != set(new_map):
        raise CvarHyperoptError("source/current raw input file sets differ")
    changed = sorted(path for path in old_map if old_map[path] != new_map[path])
    calendar_paths = [path for path in changed if path.replace("/", "\\").endswith("metadata__trading_calendar\\all.parquet")]
    disallowed = sorted(set(changed) - set(calendar_paths))
    old_schedule = pd.read_csv(old_root / "trade_day_source_map.csv")
    new_schedule = pd.read_csv(new_root / "trade_day_source_map.csv")
    schedule_match = old_schedule.equals(new_schedule)
    if disallowed or not schedule_match:
        raise CvarHyperoptError(
            f"source/current input drift is not semantically reusable; noncalendar={disallowed[:4]} schedule_match={schedule_match}"
        )
    result = {
        "status": "passed",
        "source_study_root": str(source_study_root),
        "source_rank_panel_sha256": _sha256(source_study_root / "shared_inputs" / "neutral_rank_panel.parquet"),
        "changed_raw_input_paths": changed,
        "calendar_byte_drift_allowed": bool(calendar_paths),
        "trade_day_source_map_exact_match": bool(schedule_match),
    }
    _write_json(run_root / "source_study_reuse_audit.json", result)
    return result


def prepare_inputs(
    *,
    run_root: Path,
    source_study_root: Path,
    a0_root: Path,
    b2_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    aligned, coverage, b2_status, b2_manifest = combo._load_b2_inputs(b2_root)
    dates = pd.DatetimeIndex(
        sorted(aligned.loc[aligned["strategy"].eq("equal_rank_neutral_missing"), "trade_date"])
    )
    if len(dates) != 399:
        raise CvarHyperoptError("B2 common date count is not 399")
    source_audit = _verify_score_only_a0(a0_root)
    current_audit = combo.write_as_run_input_audit(
        run_root=run_root,
        b2_root=b2_root,
        aligned=aligned,
        coverage=coverage,
        b2_status=b2_status,
        b2_manifest=b2_manifest,
    )
    reuse_audit = _compare_source_input_audits(run_root, source_study_root)
    rank_panel = _load_source_rank_panel(source_study_root, expected_dates=dates)
    base_returns = combo._b2_base_returns(aligned)
    if tuple(base_returns["trade_date"]) != tuple(dates):
        raise CvarHyperoptError("B2 base-return calendar differs from rank panel")
    pit_path = build_pit_beta_adjusted_path(run_root, base_returns)
    audit = {
        "score_only_a0": source_audit,
        "current_data_audit": current_audit,
        "source_reuse_audit": reuse_audit,
        "rank_panel_rows": int(len(rank_panel)),
        "base_return_rows": int(len(base_returns)),
        "pit_beta_path_rows": int(len(pit_path)),
    }
    _write_json(run_root / "input_preparation.json", audit)
    return rank_panel, base_returns, audit


def _source_default_cvar_output(source_study_root: Path) -> Path:
    status = _read_json(source_study_root / "run_status.json")
    record = status.get("score_method_runs", {}).get("cvar_score_weight120", {})
    output = Path(str(record.get("backtest_output", "")))
    if not output.is_dir() or not (output / "daily_returns.csv").is_file():
        raise CvarHyperoptError("source fixed-CVaR generic output is missing")
    return output


def _validate_variant_scores(scores: pd.DataFrame, *, dates: pd.DatetimeIndex, method_id: str) -> pd.DataFrame:
    return combo._validate_method_scores(scores, dates=dates, method_id=method_id)


def _load_completed_generic(record: Mapping[str, Any], *, dates: pd.DatetimeIndex, method_id: str) -> tuple[pd.DataFrame, Path] | None:
    output = Path(str(record.get("backtest_output", "")))
    if record.get("status") != "completed" or not output.is_dir() or not (output / "daily_returns.csv").is_file():
        return None
    raw = b1._load_generic_daily(output)
    return combo._validate_generic_aligned(raw, dates=dates, method_id=method_id), output


def _variant_registry_row(
    spec: CvarSpec,
    *,
    weight_digest: str,
    canonical_method_id: str,
    source_default_reuse: bool,
) -> dict[str, Any]:
    return {
        "method_id": spec.method_id,
        "family": spec.family,
        "lookback_days": spec.lookback_days,
        "tail_fraction": spec.tail_fraction,
        "lambda_cvar": spec.lambda_cvar,
        "equal_shrinkage": spec.equal_shrinkage,
        "weight_digest": weight_digest,
        "canonical_method_id": canonical_method_id,
        "is_weight_representative": bool(spec.method_id == canonical_method_id),
        "source_default_reuse": bool(source_default_reuse),
        "contract": "single_top20_score_fusion",
    }


def run_grid(
    *,
    run_root: Path,
    source_study_root: Path,
    aligned: pd.DataFrame,
    rank_panel: pd.DataFrame,
    base_returns: pd.DataFrame,
    pit_beta_path: pd.DataFrame,
    status: dict[str, Any],
) -> pd.DataFrame:
    """Run every unique CVaR parameter strategy through generic strict Top20."""

    dates = pd.DatetimeIndex(base_returns["trade_date"])
    if tuple(dates) != tuple(sorted(rank_panel["trade_date"].unique())):
        raise CvarHyperoptError("rank panel and B2 base returns have different calendars")
    b1._assert_paths_profile()
    live = b1._read_json5(b1.LIVE_CONFIG)
    strategy = b1._read_json5(b1.STRATEGY_CONFIG)
    b1._validate_execution_contract(live, strategy)
    root = run_root / "grid"
    score_root = root / "score_inputs"
    weights_root = root / "weight_audits"
    generic_root = root / "generic_backtests"
    for directory in (score_root, weights_root, generic_root):
        directory.mkdir(parents=True, exist_ok=True)
    default_output = _source_default_cvar_output(source_study_root)
    controls, control_registry = combo._b2_controls(aligned, dates=dates)
    controls = controls.loc[controls["method_id"].isin(["b2_equal_rank_baseline", "b2_regsim_standalone", "b2_p6_standalone", "b2_p3_standalone"])].copy()
    all_returns: list[pd.DataFrame] = [controls]
    summary_rows: list[dict[str, Any]] = []
    for method_id, group in controls.groupby("method_id", sort=True):
        summary_rows.extend(combo._method_summary_rows(method_id, "b2_control", group, output_path="reused_b2_aligned_output"))
    runs = status.setdefault("unique_generic_runs", {})
    by_digest: dict[str, str] = {}
    canonical_frames: dict[str, tuple[pd.DataFrame, Path]] = {}
    registry_rows: list[dict[str, Any]] = []
    specs = all_specs()
    for ordinal, spec in enumerate(specs, start=1):
        weights, audit = variant_weights(base_returns, spec, pit_beta_path=pit_beta_path)
        if spec == _default_raw_spec():
            _verify_default_parity(weights, source_study_root)
        digest = _weight_digest(weights)
        canonical_id = by_digest.setdefault(digest, spec.method_id)
        is_canonical = canonical_id == spec.method_id
        is_default = spec == _default_raw_spec()
        registry_rows.append(
            _variant_registry_row(
                spec,
                weight_digest=digest,
                canonical_method_id=canonical_id,
                source_default_reuse=is_default,
            )
        )
        audit_path = weights_root / f"{spec.method_id}.csv"
        if audit_path.exists():
            persisted = pd.read_csv(audit_path)
            if len(persisted) != len(audit):
                raise CvarHyperoptError(f"weight audit resume mismatch: {audit_path}")
        else:
            audit.to_csv(audit_path, index=False)
        if not is_canonical:
            continue
        cached = _load_completed_generic(runs.get(canonical_id, {}), dates=dates, method_id=canonical_id)
        if cached is not None:
            canonical_frames[canonical_id] = cached
            print(f"[cvar-grid] reuse completed {ordinal}/{len(specs)} {canonical_id}", flush=True)
            continue
        print(f"[cvar-grid] build {ordinal}/{len(specs)} {canonical_id}", flush=True)
        scores = combo._scores_from_rank_weights(rank_panel, weights)
        scores = _validate_variant_scores(scores, dates=dates, method_id=canonical_id)
        score_path = score_root / canonical_id / "scores.csv"
        if score_path.exists():
            stored = pd.read_csv(score_path)
            stored = _validate_variant_scores(stored, dates=dates, method_id=canonical_id)
            if len(stored) != len(scores) or not np.allclose(stored["score"], scores["score"], rtol=0.0, atol=1e-12):
                raise CvarHyperoptError(f"refusing to overwrite mismatched score input: {score_path}")
        else:
            score_path.parent.mkdir(parents=True, exist_ok=True)
            write_scores_by_date(score_path, scores, overwrite=False, dedupe=False)
        if is_default:
            source_raw = b1._load_generic_daily(default_output)
            aligned_default = combo._validate_generic_aligned(source_raw, dates=dates, method_id=canonical_id)
            output = default_output
            source_kind = "reused_verified_fixed_cvar"
        else:
            cfg = b1._generic_backtest_config(
                live=live,
                strategy=strategy,
                score_root=score_path,
                output_root=generic_root,
                start=dates[0],
                end=dates[-1],
                batch_id=f"r88_cvar_grid_{canonical_id}",
            )
            result = backtest_runtime.run(start=dates[0].date(), end=dates[-1].date(), cfg=cfg)
            raw = b1._load_generic_daily(result.out_dir)
            aligned_default = combo._validate_generic_aligned(raw, dates=dates, method_id=canonical_id)
            output = result.out_dir
            source_kind = "new_generic_strict_top20"
        canonical_frames[canonical_id] = (aligned_default, output)
        runs[canonical_id] = {
            "status": "completed",
            "method_id": canonical_id,
            "score_path": str(score_path),
            "score_sha256": _sha256(score_path),
            "weight_audit_path": str(audit_path),
            "backtest_output": str(output),
            "source_kind": source_kind,
            "raw_generic_days": int(len(aligned_default)),
            "aligned_days": int(len(aligned_default)),
            "completed_at_utc": datetime.now(timezone.utc),
        }
        _write_json(run_root / "run_status.json", status)
    registry = pd.DataFrame(registry_rows).sort_values("method_id", kind="stable")
    registry.to_csv(root / "parameter_registry.csv", index=False)
    expected_unique = int(registry["canonical_method_id"].nunique())
    if expected_unique != 61:
        raise CvarHyperoptError(f"expected 61 unique weight sequences, got {expected_unique}")
    for row in registry.itertuples(index=False):
        frame, output = canonical_frames.get(row.canonical_method_id, (None, None))
        if frame is None:
            cached = _load_completed_generic(runs.get(row.canonical_method_id, {}), dates=dates, method_id=row.canonical_method_id)
            if cached is None:
                raise CvarHyperoptError(f"missing canonical result: {row.canonical_method_id}")
            frame, output = cached
        result_frame = frame.copy()
        result_frame["method_id"] = row.method_id
        result_frame["family"] = f"cvar_{row.family}"
        all_returns.append(result_frame)
        summary_rows.extend(combo._method_summary_rows(row.method_id, f"cvar_{row.family}", result_frame, output_path=str(output)))
    combined = pd.concat(all_returns, ignore_index=True)
    expected = [*registry["method_id"].tolist(), *controls["method_id"].drop_duplicates().tolist()]
    counts = combined.groupby("method_id", sort=True).size()
    if set(counts.index) != set(expected) or not counts.eq(len(dates)).all():
        raise CvarHyperoptError("CVaR grid result coverage is incomplete")
    combined.to_csv(root / "aligned_daily_returns.csv", index=False)
    pd.DataFrame(summary_rows).sort_values(["scope", "family", "method_id"], kind="stable").to_csv(root / "summary_metrics.csv", index=False)
    status["grid_status"] = "completed"
    status["parameter_labels"] = int(len(registry))
    status["unique_generic_strategies"] = int(expected_unique)
    _write_json(run_root / "run_status.json", status)
    return combined


def _alpha_tail_rows(data: pd.DataFrame, method_ids: Sequence[str]) -> pd.DataFrame:
    """Compute beta-adjusted residual-alpha tail diagnostics by reporting scope."""

    rows: list[dict[str, Any]] = []
    for method_id in method_ids:
        frame = data.loc[data["method_id"].eq(method_id), ["trade_date", "day_return", "benchmark_return"]].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        for scope in ("development_2025", "reporting_2026", "overall"):
            subset = combo._scope(frame, scope)
            strategy = subset["day_return"].to_numpy(dtype=float)
            benchmark = subset["benchmark_return"].to_numpy(dtype=float)
            design = np.c_[np.ones(len(subset)), benchmark]
            coefficient = np.linalg.pinv(design.T @ design) @ (design.T @ strategy)
            residual = strategy - design @ coefficient
            cvar5, count5 = _tail_mean(residual, 0.05)
            cvar10, count10 = _tail_mean(residual, 0.10)
            rows.append(
                {
                    "method_id": method_id,
                    "scope": scope,
                    "alpha_daily_bp": float(coefficient[0] * 10_000.0),
                    "beta": float(coefficient[1]),
                    "alpha_residual_cvar5_bp": float(cvar5 * 10_000.0),
                    "alpha_residual_tail5_count": int(count5),
                    "alpha_residual_cvar10_bp": float(cvar10 * 10_000.0),
                    "alpha_residual_tail10_count": int(count10),
                }
            )
    return pd.DataFrame(rows)


def _weight_stability_rows(run_root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in registry.itertuples(index=False):
        path = run_root / "grid" / "weight_audits" / f"{record.method_id}.csv"
        if not path.is_file():
            raise CvarHyperoptError(f"missing weight audit: {path}")
        audit = pd.read_csv(path)
        columns = [f"weight_{model}" for model in MODELS]
        values = audit.loc[:, columns].to_numpy(dtype=float)
        l1_to_equal = np.abs(values - EQUAL_WEIGHTS).sum(axis=1)
        if len(values) > 1:
            half_l1_change = 0.5 * np.abs(np.diff(values, axis=0)).sum(axis=1)
        else:
            half_l1_change = np.array([], dtype=float)
        rows.append(
            {
                "method_id": record.method_id,
                "family": record.family,
                "mean_l1_from_equal": float(l1_to_equal.mean()),
                "max_l1_from_equal": float(l1_to_equal.max()),
                "mean_daily_weight_turnover": float(half_l1_change.mean()) if len(half_l1_change) else 0.0,
                "max_daily_weight_turnover": float(half_l1_change.max()) if len(half_l1_change) else 0.0,
                "warmup_equal_days": int(audit.get("warmup_fallback_equal", pd.Series(False)).astype(bool).sum()),
                **{f"mean_weight_{model}": float(values[:, index].mean()) for index, model in enumerate(MODELS)},
            }
        )
    return pd.DataFrame(rows)


def _selection_scorecard(
    *,
    run_root: Path,
    registry: pd.DataFrame,
    grid_data: pd.DataFrame,
    validation_root: Path,
) -> pd.DataFrame:
    summary = pd.read_csv(validation_root / "summary_metrics.csv")
    alpha_tail = _alpha_tail_rows(grid_data, registry["method_id"].tolist())
    stability = _weight_stability_rows(run_root, registry)
    development = summary.loc[summary["scope"].eq("development_2025")].copy()
    reporting = summary.loc[summary["scope"].eq("reporting_2026")].copy()
    development = development.rename(
        columns={
            "sharpe": "dev_sharpe",
            "alpha_hac_t": "dev_alpha_hac_t",
            "max_drawdown": "dev_max_drawdown",
            "cumulative_return": "dev_cumulative_return",
        }
    )
    reporting = reporting.rename(
        columns={
            "sharpe": "report_sharpe",
            "alpha_hac_t": "report_alpha_hac_t",
            "max_drawdown": "report_max_drawdown",
            "cumulative_return": "report_cumulative_return",
        }
    )
    selected_dev = development.loc[:, ["method_id", "dev_sharpe", "dev_alpha_hac_t", "dev_max_drawdown", "dev_cumulative_return"]]
    selected_report = reporting.loc[:, ["method_id", "report_sharpe", "report_alpha_hac_t", "report_max_drawdown", "report_cumulative_return"]]
    alpha_dev = alpha_tail.loc[alpha_tail["scope"].eq("development_2025")].drop(columns=["scope"])
    output = registry.merge(selected_dev, on="method_id", how="left", validate="one_to_one")
    output = output.merge(selected_report, on="method_id", how="left", validate="one_to_one")
    output = output.merge(alpha_dev, on="method_id", how="left", validate="one_to_one")
    output = output.merge(stability, on=["method_id", "family"], how="left", validate="one_to_one")
    pbo_path = validation_root / "cscv_pbo_candidates.csv"
    if pbo_path.is_file():
        pbo = pd.read_csv(pbo_path).rename(columns={"cscv_selected_count": "cscv_selected_count", "cscv_selected_oos_rank_mean": "cscv_selected_oos_rank_mean"})
        output = output.merge(pbo.loc[:, ["method_id", "cscv_selected_count", "cscv_selected_oos_rank_mean"]], on="method_id", how="left")
    return output.sort_values(["dev_sharpe", "dev_alpha_hac_t"], ascending=False, kind="stable").reset_index(drop=True)


def _write_report(run_root: Path, *, status: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    scorecard_path = run_root / "validation" / "cvar_hyperopt_union" / "selection_scorecard.csv"
    scorecard = pd.read_csv(scorecard_path) if scorecard_path.is_file() else pd.DataFrame()
    top = scorecard.head(20) if not scorecard.empty else scorecard
    lines = [
        "# R88 CVaR hyperparameter study",
        "",
        "## Contract",
        "",
        "- Frozen P6/P3/Regsim scores, current DataHub execution, o_0005, Top20, strict cycle, original fees, TWAP, benchmark, and mask are unchanged.",
        "- The raw grid has 54 labels; the residual-alpha grid has 16 labels. Exact duplicate weight sequences remain in the parameter ledger but are executed once and deduplicated for formal family tests.",
        "- 2025 is development only. 2026 is observed reporting only and is not a parameter-selection input.",
        "",
        "## Status",
        "",
        "```json",
        json.dumps(status, ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        "## Development scorecard: top 20 by 2025 Sharpe",
        "",
        combo._markdown_table(top) if not top.empty else "Not completed.",
        "",
        "## Formal family validation",
        "",
        "```json",
        json.dumps(validation, ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        "## Boundary",
        "",
        "- This is research-only. It does not authorize a live change.",
        "- 2026 results are descriptive after observation. Any deployment candidate requires a frozen configuration and fresh forward shadow evidence.",
    ]
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_validation(run_root: Path, status: dict[str, Any]) -> dict[str, Any]:
    grid_root = run_root / "grid"
    data_path = grid_root / "aligned_daily_returns.csv"
    registry_path = grid_root / "parameter_registry.csv"
    if not data_path.is_file() or not registry_path.is_file():
        raise CvarHyperoptError("grid outputs are missing before validation")
    data = pd.read_csv(data_path)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    registry = pd.read_csv(registry_path)
    method_ids = [*registry["method_id"].tolist(), "b2_equal_rank_baseline", "b2_regsim_standalone", "b2_p6_standalone", "b2_p3_standalone"]
    validation = combo._family_validation(
        run_root=run_root,
        data=data,
        method_ids=method_ids,
        reference_id="b2_equal_rank_baseline",
        family="cvar_hyperopt_union",
        include_static_pbo=True,
    )
    validation_root = run_root / "validation" / "cvar_hyperopt_union"
    scorecard = _selection_scorecard(
        run_root=run_root,
        registry=registry,
        grid_data=data,
        validation_root=validation_root,
    )
    scorecard.to_csv(validation_root / "selection_scorecard.csv", index=False)
    status["validation_status"] = "completed"
    _write_json(run_root / "run_status.json", status)
    _write_report(run_root, status=status, validation=validation)
    return validation


def run_study(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str = DEFAULT_RUN_NAME,
    source_study_root: Path = DEFAULT_SOURCE_STUDY_ROOT,
    a0_root: Path = DEFAULT_A0_ROOT,
    b2_root: Path = DEFAULT_B2_ROOT,
    stage: str = "all",
    resume: bool = False,
) -> dict[str, Any]:
    run_root = _assert_run_root(output_root, run_name, resume=resume)
    status_path = run_root / "run_status.json"
    if status_path.is_file():
        if not resume:
            raise CvarHyperoptError("existing study state requires explicit --resume")
        status = _read_json(status_path)
        expected = {
            "source_study_root": str(source_study_root),
            "a0_root": str(a0_root),
            "b2_root": str(b2_root),
        }
        if any(str(status.get(key, "")) != value for key, value in expected.items()):
            raise CvarHyperoptError("refusing to resume with different frozen inputs")
    else:
        status = _initial_status(run_root, source_study_root=source_study_root, a0_root=a0_root, b2_root=b2_root)
        _write_json(status_path, status)
    b1._assert_paths_profile()
    rank_panel, base_returns, input_audit = prepare_inputs(
        run_root=run_root,
        source_study_root=source_study_root,
        a0_root=a0_root,
        b2_root=b2_root,
    )
    pit_path = build_pit_beta_adjusted_path(run_root, base_returns)
    status["input_preparation"] = input_audit
    status["prepare_status"] = "completed"
    status["pit_beta_path_rows"] = int(len(pit_path))
    _write_json(status_path, status)
    if stage == "prepare":
        return status
    aligned, _, _, _ = combo._load_b2_inputs(b2_root)
    if stage in {"grid", "all"}:
        combo.verify_as_run_input_audit(run_root=run_root, checkpoint="before_grid")
        run_grid(
            run_root=run_root,
            source_study_root=source_study_root,
            aligned=aligned,
            rank_panel=rank_panel,
            base_returns=base_returns,
            pit_beta_path=pit_path,
            status=status,
        )
    if stage in {"validate", "all"}:
        combo.verify_as_run_input_audit(run_root=run_root, checkpoint="before_validation")
        run_validation(run_root, status)
    complete = status.get("grid_status") == "completed" and status.get("validation_status") == "completed"
    status["status"] = "completed" if complete else f"completed_{stage}"
    status["completed_at_utc"] = datetime.now(timezone.utc) if complete else status.get("completed_at_utc")
    _write_json(status_path, status)
    return status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--source-study-root", type=Path, default=DEFAULT_SOURCE_STUDY_ROOT)
    parser.add_argument("--a0-root", type=Path, default=DEFAULT_A0_ROOT)
    parser.add_argument("--b2-root", type=Path, default=DEFAULT_B2_ROOT)
    parser.add_argument("--stage", choices=("prepare", "grid", "validate", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_study(
        output_root=args.output_root,
        run_name=str(args.run_name),
        source_study_root=args.source_study_root,
        a0_root=args.a0_root,
        b2_root=args.b2_root,
        stage=str(args.stage),
        resume=bool(args.resume),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


def _initial_status(run_root: Path, *, source_study_root: Path, a0_root: Path, b2_root: Path) -> dict[str, Any]:
    return {
        "schema": "r88_cvar_hyperopt/v1",
        "status": "running",
        "research_only": True,
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "model_training_called": False,
        "model_scoring_called": False,
        "run_root": str(run_root),
        "source_study_root": str(source_study_root),
        "a0_root": str(a0_root),
        "b2_root": str(b2_root),
        "grid": {"raw_labels": len(raw_specs()), "residual_labels": len(residual_specs()), "total_labels": len(all_specs())},
        "started_at_utc": datetime.now(timezone.utc),
    }


if __name__ == "__main__":
    main()
