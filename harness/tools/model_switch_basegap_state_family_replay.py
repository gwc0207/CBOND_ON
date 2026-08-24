"""Research-only BaseGap state-family replay on frozen live50 inputs.

Stage A already selected the BaseGap *internal* parameter point using only the
design period.  This separate, bounded Stage B keeps that point fixed and
compares four preregistered state families:

1. frozen 44-column path state only;
2. path state plus all 19 frozen current/previous-score geometry features;
3. path state plus a causal 12-column compact summary of frozen live50 factor
   values and their cross-sectional correlation structure;
4. all three blocks together.

The output is entirely under ``research_scratch``.  It never writes or reads
the live runtime, live configuration, DB, scheduler, production state, or
production result directories.  Candidate selection remains symmetric direct
argmax among Regsim, Ensemble, and HL20: no Champion preference, Robust,
Fusion, threshold, or gate is present.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json5
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cbond_on.infra.live.model_switch import _scoreopt_score_sample
from harness.tools.model_switch_basegap_tuning_replay import (
    DEFAULT_SOURCE_RUN,
    MODELS,
    SPLITS,
    _scope_frame,
    _scope_metrics,
    parity_audit,
    prepare_panel,
    sha256,
    validate_frozen_source,
)


DEFAULT_STAGE_A_RUN = Path(
    "D:/cbond_on/research_scratch/model_switch_basegap_tuning_20260811/"
    "run_stage_a_20260811_r3"
)
DEFAULT_FACTOR_FREEZE_ROOT = Path(
    "D:/cbond_on/research_scratch/live50_feature_model_retuning_20260807_r1/"
    "inputs/frozen_inputs_20260807_r1"
)
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\model_switch_basegap_state_family_20260811")
FACTOR_CONTRACT_PATH = REPO_ROOT / "cbond_on" / "config" / "models" / "lgbm" / "lgbm_live50_feature_contract_20260805_config.json5"

FIXED_LOOKBACK_DAYS = 240
FIXED_NEAREST_K = 60
FIXED_MIN_PERIODS = 60
FIXED_METRIC = "trim20_lcb10"
FACTOR_TEMPORAL_LOOKBACK = 120
FACTOR_TEMPORAL_MIN_PERIODS = 60
LEGACY_FACTOR_COUNT = 27


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_inside(path: Path, parent: Path, *, description: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} must be under {root}: {resolved}") from exc
    return resolved


def _assert_output_root(path: Path) -> Path:
    return _resolve_inside(path, DEFAULT_OUTPUT_ROOT, description="research output")


def _make_run_root(output_root: Path, run_name: str | None) -> Path:
    root = _assert_output_root(output_root)
    name = run_name or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"run name must be one plain path component, got {name!r}")
    run_root = _assert_output_root(root / name)
    if run_root == DEFAULT_OUTPUT_ROOT.resolve() or run_root.exists():
        raise FileExistsError(f"refusing to overwrite research output: {run_root}")
    return run_root


def _assert_locked_source(path: Path, expected: Path, *, name: str, required: Sequence[str]) -> Path:
    resolved = path.resolve()
    if resolved != expected.resolve():
        raise ValueError(f"{name} is locked to {expected.resolve()}, got {resolved}")
    for relative in required:
        if not (resolved / relative).exists():
            raise FileNotFoundError(f"{name} is incomplete; missing {relative}: {resolved}")
    return resolved


def _load_json5(path: Path) -> dict[str, Any]:
    value = json5.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping config: {path}")
    return value


def load_factor_contract() -> tuple[str, ...]:
    config = _load_json5(FACTOR_CONTRACT_PATH)
    factors = config.get("factors")
    if not isinstance(factors, list) or len(factors) != 50 or not all(isinstance(item, str) for item in factors):
        raise ValueError(f"expected exactly ordered 50-factor contract: {FACTOR_CONTRACT_PATH}")
    if len(set(factors)) != 50:
        raise ValueError("factor contract contains duplicate feature names")
    return tuple(factors)


def _factor_store_entry(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("factor freeze manifest has no entries")
    matches = [item for item in entries if isinstance(item, Mapping) and item.get("name") == "factor_store_live50"]
    if len(matches) != 1:
        raise ValueError("factor freeze manifest must contain exactly one factor_store_live50 entry")
    return matches[0]


def verify_factor_freeze(score_days: Sequence[object]) -> tuple[Path, dict[str, object]]:
    """Verify every frozen factor file used by this replay against its manifest."""

    freeze_root = _assert_locked_source(
        DEFAULT_FACTOR_FREEZE_ROOT,
        DEFAULT_FACTOR_FREEZE_ROOT,
        name="factor freeze",
        required=("manifest.json", "factor_data/factors/T1430"),
    )
    manifest_path = freeze_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "passed":
        raise ValueError(f"factor freeze is not passed: {manifest.get('status')!r}")
    entry = _factor_store_entry(manifest)
    factor_root = _resolve_inside(Path(str(entry.get("destination", ""))), freeze_root, description="frozen factor store")
    expected_root = (freeze_root / "factor_data").resolve()
    if factor_root != expected_root:
        raise ValueError(f"factor freeze destination drift: {factor_root}, expected {expected_root}")
    inventory = entry.get("destination_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("factor freeze entry has no destination inventory")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("factor freeze inventory has no files")
    hash_by_relative = {
        str(item.get("relative_path")): str(item.get("sha256"))
        for item in files
        if isinstance(item, Mapping) and item.get("relative_path") and item.get("sha256")
    }
    audit_rows: list[dict[str, object]] = []
    for raw_day in score_days:
        day = pd.Timestamp(raw_day).date()
        relative = Path("factors") / "T1430" / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
        path = factor_root / relative
        expected_hash = hash_by_relative.get(relative.as_posix())
        if not expected_hash or not path.is_file():
            raise FileNotFoundError(f"frozen factor file missing from manifest/store: {relative}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"frozen factor SHA-256 mismatch: {path}")
        audit_rows.append(
            {
                "score_day": str(day),
                "relative_path": relative.as_posix(),
                "sha256": actual_hash,
                "bytes": int(path.stat().st_size),
            }
        )
    verification = {
        "factor_freeze_root": str(freeze_root),
        "factor_manifest": str(manifest_path),
        "factor_manifest_sha256": sha256(manifest_path),
        "factor_tree_sha256": str(inventory.get("tree_sha256")),
        "verified_used_factor_files": len(audit_rows),
        "verified_used_total_bytes": int(sum(int(item["bytes"]) for item in audit_rows)),
        "files": audit_rows,
    }
    return factor_root, verification


def _score_path(input_root: Path, model: str, score_day: date) -> Path:
    return input_root / "scores" / model / f"{score_day:%Y-%m}" / f"{score_day:%Y-%m-%d}.csv"


def load_common_score_codes(input_root: Path, score_day: date) -> list[str]:
    """Return ordered frozen score-universe codes after strict three-way audit."""

    ordered_codes: list[str] | None = None
    expected_set: set[str] | None = None
    for model in MODELS:
        path = _score_path(input_root, model, score_day)
        frame = pd.read_csv(path, usecols=["code", "score"])
        frame["code"] = frame["code"].astype(str)
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        frame = frame.dropna(subset=["code", "score"]).drop_duplicates("code", keep="last")
        if frame.empty:
            raise ValueError(f"empty frozen score universe: {path}")
        codes = frame["code"].tolist()
        code_set = set(codes)
        if expected_set is None:
            expected_set = code_set
            ordered_codes = codes
        elif code_set != expected_set:
            raise ValueError(f"three candidate score universes differ on {score_day}: {model}")
    if ordered_codes is None:
        raise AssertionError("no candidate scores loaded")
    return ordered_codes


def _factor_path(factor_root: Path, score_day: date) -> Path:
    return factor_root / "factors" / "T1430" / f"{score_day:%Y-%m}" / f"{score_day:%Y%m%d}.parquet"


def load_factor_cross_section(
    *,
    factor_root: Path,
    score_day: date,
    score_codes: Sequence[str],
    factor_names: Sequence[str],
) -> pd.DataFrame:
    """Read one frozen 14:30 factor slice and align it only to score codes."""

    path = _factor_path(factor_root, score_day)
    frame = pd.read_parquet(path, columns=list(factor_names))
    if list(frame.columns) != list(factor_names):
        raise ValueError(f"frozen factor order differs from the live50 contract: {path}")
    if not isinstance(frame.index, pd.MultiIndex) or "dt" not in frame.index.names or "code" not in frame.index.names:
        raise ValueError(f"frozen factor index must be (dt, code): {path}")
    timestamp_days = pd.to_datetime(frame.index.get_level_values("dt"), errors="coerce").date
    if set(timestamp_days) != {score_day}:
        raise ValueError(f"factor timestamp/date mismatch for {path}: {set(timestamp_days)} vs {score_day}")
    codes = frame.index.get_level_values("code").astype(str)
    if codes.duplicated().any():
        raise ValueError(f"duplicate factor code in frozen slice: {path}")
    missing_codes = set(str(code) for code in score_codes) - set(codes)
    if missing_codes:
        raise ValueError(f"frozen factor slice lacks score-universe codes ({len(missing_codes)}): {path}")
    aligned = frame.copy()
    aligned.index = pd.Index(codes, name="code")
    aligned = aligned.reindex(pd.Index(list(score_codes), name="code"))
    if aligned.index.isna().any() or len(aligned) != len(score_codes):
        raise ValueError(f"unable to align factor rows to frozen score codes: {path}")
    return aligned.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _mean_upper(correlation: pd.DataFrame, left: Sequence[str], right: Sequence[str], *, same_block: bool) -> float:
    values: list[float] = []
    if same_block:
        for index, left_name in enumerate(left):
            for right_name in left[index + 1 :]:
                value = correlation.at[left_name, right_name]
                if math.isfinite(float(value)):
                    values.append(abs(float(value)))
    else:
        for left_name in left:
            for right_name in right:
                value = correlation.at[left_name, right_name]
                if math.isfinite(float(value)):
                    values.append(abs(float(value)))
    return float(np.mean(values)) if values else float("nan")


def daily_factor_statistics(frame: pd.DataFrame, factor_names: Sequence[str]) -> tuple[dict[str, float], pd.Series, pd.Series]:
    """Compute current-day, return-free factor structure statistics.

    The returned per-factor medians and log-IQR values are later time-scaled
    with strictly earlier dates.  No realised candidate return enters here.
    """

    legacy = list(factor_names[:LEGACY_FACTOR_COUNT])
    mined = list(factor_names[LEGACY_FACTOR_COUNT:])
    if len(legacy) != LEGACY_FACTOR_COUNT or len(mined) != 23:
        raise ValueError("expected 27 legacy and 23 mined factors")
    numeric = frame[list(factor_names)].replace([np.inf, -np.inf], np.nan)
    medians = numeric.median(axis=0, skipna=True)
    q10 = numeric.quantile(0.10, axis=0)
    q25 = numeric.quantile(0.25, axis=0)
    q75 = numeric.quantile(0.75, axis=0)
    q90 = numeric.quantile(0.90, axis=0)
    iqr = q75 - q25
    denom = q90 - q10
    tail_asymmetry = ((q90 - medians) - (medians - q10)) / denom.where(denom.abs() > 1e-12)
    log_iqr = np.log(iqr.where(iqr > 1e-12))
    ranks = numeric.rank(axis=0, method="average", na_option="keep")
    correlation = ranks.corr(method="pearson", min_periods=80)
    upper_values = [
        abs(float(correlation.at[left, right]))
        for index, left in enumerate(factor_names)
        for right in factor_names[index + 1 :]
        if math.isfinite(float(correlation.at[left, right]))
    ]
    stats = {
        "factor_universe_n": float(len(numeric)),
        "factor_cell_missing_rate": float(numeric.isna().to_numpy().mean()),
        "factor_tail_asymmetry": float(tail_asymmetry.median(skipna=True)),
        "factor_absrho_mean_all": float(np.mean(upper_values)) if upper_values else float("nan"),
        "factor_absrho_p90_all": float(np.quantile(upper_values, 0.90)) if upper_values else float("nan"),
        "factor_absrho_mean_legacy": _mean_upper(correlation, legacy, legacy, same_block=True),
        "factor_absrho_mean_mined": _mean_upper(correlation, mined, mined, same_block=True),
        "factor_absrho_mean_cross_legacy_mined": _mean_upper(correlation, legacy, mined, same_block=False),
    }
    return stats, medians, log_iqr


def causal_robust_z(raw: pd.DataFrame, *, lookback: int, min_periods: int) -> pd.DataFrame:
    """Strictly-predecessor rolling median/MAD standardisation."""

    if lookback <= 0 or min_periods <= 0:
        raise ValueError("lookback and min_periods must be positive")
    output = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns, dtype=float)
    for position in range(len(raw)):
        history = raw.iloc[max(0, position - lookback) : position]
        current = raw.iloc[position]
        for column in raw.columns:
            values = pd.to_numeric(history[column], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) < min_periods or not math.isfinite(float(current[column])):
                continue
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            scale = 1.4826 * mad
            if math.isfinite(scale) and scale > 1e-12:
                output.iat[position, output.columns.get_loc(column)] = (float(current[column]) - center) / scale
    return output


def build_factor_state(
    *,
    input_root: Path,
    factor_root: Path,
    score_days: Sequence[object],
    factor_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build F12 from frozen T1430 factors and frozen score universes only."""

    raw_rows: list[dict[str, object]] = []
    median_rows: list[pd.Series] = []
    log_iqr_rows: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []
    for raw_day in score_days:
        score_day = pd.Timestamp(raw_day).date()
        score_codes = load_common_score_codes(input_root, score_day)
        cross_section = load_factor_cross_section(
            factor_root=factor_root,
            score_day=score_day,
            score_codes=score_codes,
            factor_names=factor_names,
        )
        stats, medians, log_iqr = daily_factor_statistics(cross_section, factor_names)
        raw_rows.append({"score_day": score_day, **stats})
        median_rows.append(medians.rename(score_day))
        log_iqr_rows.append(log_iqr.rename(score_day))
        audit_rows.append(
            {
                "score_day": score_day,
                "score_universe_n": len(score_codes),
                "factor_rows_before_score_alignment": None,
                "factor_rows_after_score_alignment": len(cross_section),
                "factor_cell_missing_rate": stats["factor_cell_missing_rate"],
                "factor_file": str(_factor_path(factor_root, score_day)),
            }
        )
    raw = pd.DataFrame(raw_rows).sort_values("score_day").reset_index(drop=True)
    median_frame = pd.DataFrame(median_rows).reindex(columns=list(factor_names)).reset_index(drop=True)
    log_iqr_frame = pd.DataFrame(log_iqr_rows).reindex(columns=list(factor_names)).reset_index(drop=True)
    z_medians = causal_robust_z(median_frame, lookback=FACTOR_TEMPORAL_LOOKBACK, min_periods=FACTOR_TEMPORAL_MIN_PERIODS)
    z_log_iqr = causal_robust_z(log_iqr_frame, lookback=FACTOR_TEMPORAL_LOOKBACK, min_periods=FACTOR_TEMPORAL_MIN_PERIODS)
    raw["factor_loc_legacy"] = z_medians.iloc[:, :LEGACY_FACTOR_COUNT].median(axis=1, skipna=True)
    raw["factor_loc_mined"] = z_medians.iloc[:, LEGACY_FACTOR_COUNT:].median(axis=1, skipna=True)
    raw["factor_scale_legacy"] = z_log_iqr.iloc[:, :LEGACY_FACTOR_COUNT].median(axis=1, skipna=True)
    raw["factor_scale_mined"] = z_log_iqr.iloc[:, LEGACY_FACTOR_COUNT:].median(axis=1, skipna=True)
    ordered = raw[
        [
            "score_day",
            "factor_universe_n",
            "factor_cell_missing_rate",
            "factor_loc_legacy",
            "factor_loc_mined",
            "factor_scale_legacy",
            "factor_scale_mined",
            "factor_tail_asymmetry",
            "factor_absrho_mean_all",
            "factor_absrho_p90_all",
            "factor_absrho_mean_legacy",
            "factor_absrho_mean_mined",
            "factor_absrho_mean_cross_legacy_mined",
        ]
    ].copy()
    return ordered, pd.DataFrame(audit_rows)


def run_block_basegap_variant(
    panel: pd.DataFrame,
    *,
    blocks: Mapping[str, Sequence[str]],
    variant: str,
    lookback_days: int = FIXED_LOOKBACK_DAYS,
    nearest_k: int = FIXED_NEAREST_K,
    min_periods: int = FIXED_MIN_PERIODS,
    metric: str = FIXED_METRIC,
) -> pd.DataFrame:
    """BaseGap replay with path-scale-preserving equal block contributions."""

    if lookback_days <= 0 or nearest_k <= 0 or min_periods <= 0 or nearest_k > lookback_days:
        raise ValueError("invalid fixed BaseGap parameter point")
    block_items = [(name, list(columns)) for name, columns in blocks.items()]
    if not block_items or any(not columns for _, columns in block_items):
        raise ValueError("at least one non-empty state block is required")
    features = [column for _, columns in block_items for column in columns]
    if len(set(features)) != len(features):
        raise ValueError("state features cannot appear in more than one block")
    missing = [column for column in ["score_day", *MODELS, *features] if column not in panel.columns]
    if missing:
        raise KeyError(f"state-family panel missing columns: {missing}")
    reference_width = len(block_items[0][1])
    ordered = panel.sort_values("score_day").reset_index(drop=True).copy()
    if "state_available" not in ordered.columns:
        ordered["state_available"] = True
    ordered["state_available"] = ordered["state_available"].fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    for position, current in ordered.iterrows():
        score_day = pd.Timestamp(current["score_day"]).date()
        returns_history = ordered.iloc[:position][["score_day", *MODELS]]
        history_end = returns_history["score_day"].max() if not returns_history.empty else None
        record: dict[str, object] = {
            "score_day": score_day,
            "variant": variant,
            "lookback_days": int(lookback_days),
            "nearest_k": int(nearest_k),
            "min_periods": int(min_periods),
            "metric": metric,
            "basegap_history_end": history_end,
            "basegap_history_days": 0,
            "basegap_reason": None,
            "basegap_selected_name": "Regsim",
            "basegap_score_gap": float("nan"),
            "neighbor_min_day": None,
            "neighbor_max_day": None,
            "neighbor_count": 0,
        }
        for model in MODELS:
            record[f"basegap_score_{model}"] = float("nan")
        current_values = current[features].to_numpy(dtype=float)
        if not bool(current["state_available"]):
            record["basegap_reason"] = "feature_missing"
        elif not np.isfinite(current_values).all():
            record["basegap_reason"] = "feature_na"
        else:
            joined = ordered.iloc[:position][["score_day", "state_available", *MODELS, *features]].copy()
            joined = joined.loc[joined["state_available"]].tail(lookback_days).copy()
            history_features = joined[features]
            history_returns = joined[list(MODELS)]
            valid_mask = ~(history_features.isna().any(axis=1) | history_returns.isna().any(axis=1))
            history_features = history_features.loc[valid_mask]
            history_returns = history_returns.loc[valid_mask]
            observations = int(len(history_features))
            record["basegap_history_days"] = observations
            if observations < nearest_k or observations < min_periods:
                record["basegap_reason"] = "insufficient_history"
            else:
                mean = history_features.mean(axis=0)
                std = history_features.std(axis=0).replace(0, math.nan).fillna(1.0)
                distance_squared = pd.Series(0.0, index=history_features.index)
                for _, columns in block_items:
                    history_z = (history_features[columns] - mean[columns]) / std[columns]
                    current_z = (current[columns] - mean[columns]) / std[columns]
                    # Each block gets the same aggregate importance as the
                    # original 44-column path block.  With path alone this is
                    # exactly production Euclidean distance, not merely a
                    # monotone rescaling.
                    block_scale = reference_width / len(columns)
                    distance_squared = distance_squared + block_scale * (history_z.sub(current_z, axis=1) ** 2).sum(axis=1)
                distances = distance_squared.pow(0.5)
                nearest_index = distances.sort_values().index[:nearest_k]
                sample = history_returns.loc[nearest_index]
                scores = _scoreopt_score_sample(sample, model_cols=list(MODELS), score_mode=metric).sort_values(ascending=False)
                selected = str(scores.index[0])
                best_score = float(scores.iloc[0])
                second_score = float(scores.iloc[1]) if len(scores) > 1 else float("nan")
                score_gap = best_score - second_score if math.isfinite(second_score) else float("nan")
                neighbour_days = pd.to_datetime(joined.loc[nearest_index, "score_day"], errors="coerce").dt.date
                record.update(
                    {
                        "basegap_selected_name": selected,
                        "basegap_reason": "score_best" if score_gap > 0.0 else "margin_default",
                        "basegap_history_days": int(len(sample)),
                        "basegap_score_gap": score_gap,
                        "neighbor_min_day": neighbour_days.min(),
                        "neighbor_max_day": neighbour_days.max(),
                        "neighbor_count": int(len(sample)),
                    }
                )
                for model in MODELS:
                    record[f"basegap_score_{model}"] = float(scores[model])
        selected_name = str(record["basegap_selected_name"])
        realised = {model: float(current[model]) for model in MODELS}
        selected_return = realised[selected_name]
        best_name = max(MODELS, key=lambda model: realised[model])
        record.update(
            {
                **{f"realized_return_{model}": realised[model] for model in MODELS},
                "selected_return": selected_return,
                "equal_weight_return": float(np.mean(list(realised.values()))),
                "best_name": best_name,
                "best_return": realised[best_name],
                "selection_alpha": selected_return - float(np.mean(list(realised.values()))),
                "regret": realised[best_name] - selected_return,
                "selected_rank": int(1 + sum(realised[model] > selected_return for model in MODELS)),
                "selection_active": bool(record["basegap_reason"] in {"score_best", "margin_default"}),
                "execution_metadata_complete": bool(current.get("execution_metadata_complete", False)),
                "state_block_count": len(block_items),
                "state_feature_count": len(features),
            }
        )
        rows.append(record)
    result = pd.DataFrame(rows)
    active = result.loc[result["selection_active"]]
    if not active.empty and not (pd.to_datetime(active["neighbor_max_day"]).dt.date < pd.to_datetime(active["score_day"]).dt.date).all():
        raise RuntimeError("state-family replay leakage: neighbour is not strictly before score_day")
    return result


def _families(path_features: Sequence[str], geometry_features: Sequence[str], factor_features: Sequence[str]) -> list[dict[str, object]]:
    return [
        {
            "variant": "path44_only",
            "family": "path_only",
            "blocks": {"path44": list(path_features)},
            "lookback_days": FIXED_LOOKBACK_DAYS,
            "nearest_k": FIXED_NEAREST_K,
            "min_periods": FIXED_MIN_PERIODS,
            "metric": FIXED_METRIC,
            "is_stage_a_reference": True,
        },
        {
            "variant": "path44_plus_score_geometry_g19",
            "family": "path_plus_g19",
            "blocks": {"path44": list(path_features), "score_geometry_g19": list(geometry_features)},
            "lookback_days": FIXED_LOOKBACK_DAYS,
            "nearest_k": FIXED_NEAREST_K,
            "min_periods": FIXED_MIN_PERIODS,
            "metric": FIXED_METRIC,
            "is_stage_a_reference": False,
        },
        {
            "variant": "path44_plus_factor_structure_f12",
            "family": "path_plus_f12",
            "blocks": {"path44": list(path_features), "factor_structure_f12": list(factor_features)},
            "lookback_days": FIXED_LOOKBACK_DAYS,
            "nearest_k": FIXED_NEAREST_K,
            "min_periods": FIXED_MIN_PERIODS,
            "metric": FIXED_METRIC,
            "is_stage_a_reference": False,
        },
        {
            "variant": "path44_plus_factor_structure_f12_plus_score_geometry_g19",
            "family": "path_plus_f12_plus_g19",
            "blocks": {
                "path44": list(path_features),
                "factor_structure_f12": list(factor_features),
                "score_geometry_g19": list(geometry_features),
            },
            "lookback_days": FIXED_LOOKBACK_DAYS,
            "nearest_k": FIXED_NEAREST_K,
            "min_periods": FIXED_MIN_PERIODS,
            "metric": FIXED_METRIC,
            "is_stage_a_reference": False,
        },
    ]


def _family_summary(all_daily: pd.DataFrame, families: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str | None, str | None]] = [("full", None, None), *SPLITS]
    for family in families:
        daily = all_daily.loc[all_daily["variant"] == family["variant"]].copy()
        metadata = {key: value for key, value in family.items() if key != "blocks"}
        metadata["state_blocks"] = "+".join(family["blocks"].keys())
        metadata["state_feature_count"] = sum(len(item) for item in family["blocks"].values())
        for scope, start, end in scopes:
            rows.append(_scope_metrics(_scope_frame(daily, start=start, end=end, active_only=False), variant_row=metadata, scope=scope, active_only=False))
            rows.append(_scope_metrics(_scope_frame(daily, start=start, end=end, active_only=True), variant_row=metadata, scope=scope, active_only=True))
    return pd.DataFrame(rows)


def rank_design_families(summary: pd.DataFrame) -> pd.DataFrame:
    ranking = summary.loc[(summary["scope"] == "design") & summary["active_only"]].copy()
    ranking = ranking.loc[ranking["n_days"] > 0].copy()
    ranking = ranking.sort_values(
        ["selection_alpha_mean_bp", "hit_rate", "mean_regret_bp", "state_feature_count", "variant"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "design_rank", np.arange(1, len(ranking) + 1))
    return ranking


def _plot_state_summary(run_root: Path, summary: pd.DataFrame) -> None:
    scopes = ("design", "validation", "final_oos")
    plot = summary.loc[summary["active_only"] & summary["scope"].isin(scopes)].copy()
    variants = plot["variant"].drop_duplicates().tolist()
    figure, axis = plt.subplots(figsize=(13, 6))
    x = np.arange(len(variants))
    width = 0.23
    for offset, scope in enumerate(scopes):
        values = [
            float(plot.loc[(plot["variant"] == variant) & (plot["scope"] == scope), "selection_alpha_mean_bp"].iloc[0])
            for variant in variants
        ]
        axis.bar(x + (offset - 1) * width, values, width=width, label=scope)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels=variants, rotation=16, ha="right")
    axis.set_ylabel("selection alpha vs arithmetic equal weight (bp/day)")
    axis.set_title("Preregistered state-family comparison; OOS is report-only")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_root / "state_family_selection_alpha.png", dpi=160)
    plt.close(figure)


def _task_state(*, phase: str, run_root: Path, source_run: Path) -> str:
    return f"""# Task State

## Objective

- Frozen, research-only comparison of four preregistered pure-BaseGap state families after the Stage-A fixed parameter selection.

## Risk Level

- medium: state-family selection may overfit; all runtime and live assets are read-only.

## Current Verified Facts

- selector source: `{source_run}`
- factor source: `{DEFAULT_FACTOR_FREEZE_ROOT}`
- output: `{run_root}`
- no DB, scheduler, live runtime, production config, model state, factor store, or live result path is written.

## Files Changed

- only files under this isolated research run root.

## Open Risks

- score geometry is based on the existing T1430 score contract; it is not new strict-14:29 PIT evidence.
- this is selector-shadow-return research, not a new full single-book execution replay.

## Next Action

- {"completed; do not select again using final OOS" if phase == "completed" else "verify frozen source/factor hashes, compute F12, parity-check path44, then run four fixed families"}.

## Handoff Summary

- phase: {phase}
"""


def _write_results(run_root: Path, *, source_run: Path, ranking: pd.DataFrame, summary: pd.DataFrame) -> None:
    winner = ranking.iloc[0]
    report = summary.loc[
        summary["variant"].isin(["path44_only", str(winner["variant"])])
        & summary["active_only"]
        & summary["scope"].isin(["design", "validation", "final_oos", "full"])
    ].sort_values(["variant", "scope"])
    lines = [
        "# Frozen pure-BaseGap state-family replay", "",
        "## Scope", "",
        f"- Frozen selector input: `{source_run}`",
        f"- Fixed parameters selected in Stage A design only: lookback={FIXED_LOOKBACK_DAYS}, K={FIXED_NEAREST_K}, metric={FIXED_METRIC}.",
        "- Four whole state families were preregistered: path44, path44+G19, path44+F12, path44+F12+G19.",
        "- Blocks receive equal aggregate distance importance; the path-only implementation is an exact Stage-A parity reference.",
        "- No threshold, Champion, Robust, Fusion, return-derived same-day feature, factor/model/strategy change, or live mutation exists.", "",
        "## Design-only state-family winner", "",
        f"- `{winner['variant']}`; design active selection alpha={float(winner['selection_alpha_mean_bp']):.3f} bp/day.",
        "- Validation and final OOS are report-only. They must not be used to choose another family.", "",
        "## Aligned summary", "",
        "| variant | scope | n | alpha (bp/day) | hit rate | mean regret (bp) | selected Sharpe | selected MDD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in report.iterrows():
        lines.append(
            "| {variant} | {scope} | {n} | {alpha:.3f} | {hit:.2%} | {regret:.3f} | {sharpe} | {mdd} |".format(
                variant=row["variant"],
                scope=row["scope"],
                n=int(row["n_days"]),
                alpha=float(row["selection_alpha_mean_bp"]),
                hit=float(row["hit_rate"]),
                regret=float(row["mean_regret_bp"]),
                sharpe="" if pd.isna(row["selected_sharpe"]) else f"{float(row['selected_sharpe']):.3f}",
                mdd="" if pd.isna(row["selected_max_drawdown"]) else f"{float(row['selected_max_drawdown']):.2%}",
            )
        )
    lines.extend(
        [
            "", "## F12 definition", "",
            "- score-universe count and factor-cell missing rate; causal median/MAD-normalized cross-sectional location and log-IQR summaries for legacy-27 and mined-23 factors; tail asymmetry; and five Spearman correlation-structure summaries.",
            "- F12 uses only each score day's frozen 14:30 factor cross-section and strictly prior factor days for its temporal normalisation. It never uses candidate returns, label/IC, or later inputs.",
            "", "## Evidence", "",
            "- `path44_stage_a_parity_audit.csv`, `factor_state_v1.csv`, `factor_input_audit.csv`, `daily_state_family_replay.csv`, `summary_metrics.csv`, `design_ranking.csv`, `leakage_audit.json`, and `state_family_selection_alpha.png`.",
            "",
        ]
    )
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_replay(*, source_run: Path, stage_a_run: Path, run_root: Path) -> dict[str, object]:
    source_run = _assert_locked_source(source_run, DEFAULT_SOURCE_RUN, name="selector source", required=("run_manifest.json", "input_snapshot", "daily_score_disagreement.csv"))
    stage_a_run = _assert_locked_source(stage_a_run, DEFAULT_STAGE_A_RUN, name="Stage-A source", required=("run_status.json", "daily_selector_replay.csv", "run_manifest.json"))
    stage_status = json.loads((stage_a_run / "run_status.json").read_text(encoding="utf-8"))
    if stage_status.get("status") != "completed":
        raise ValueError("Stage A must have completed before state-family replay")
    run_root = _assert_output_root(run_root)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite research run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "task_state.md").write_text(_task_state(phase="running", run_root=run_root, source_run=source_run), encoding="utf-8")
    _write_json(
        run_root / "run_status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc, ), "database_writes": False, "live_runtime_called": False, "scheduler_called": False, "research_output_only": True},
    )

    config, selector_verification = validate_frozen_source(source_run)
    panel, path_features = prepare_panel(config)
    input_root = source_run / "input_snapshot"
    factor_names = load_factor_contract()
    factor_root, factor_verification = verify_factor_freeze(panel["score_day"].tolist())
    factor_state, factor_audit = build_factor_state(
        input_root=input_root,
        factor_root=factor_root,
        score_days=panel["score_day"].tolist(),
        factor_names=factor_names,
    )
    geometry = pd.read_csv(source_run / "daily_score_disagreement.csv")
    geometry["score_day"] = pd.to_datetime(geometry["score_day"], errors="coerce").dt.date
    geometry = geometry.dropna(subset=["score_day"]).drop_duplicates("score_day", keep="last")
    geometry_features = tuple(column for column in geometry.columns if column != "score_day")
    if len(geometry_features) != 19:
        raise ValueError(f"expected frozen G19 score geometry, got {len(geometry_features)} columns")
    factor_features = tuple(column for column in factor_state.columns if column != "score_day")
    if len(factor_features) != 12:
        raise ValueError(f"expected compact F12 factor state, got {len(factor_features)} columns")
    enriched = panel.merge(geometry, on="score_day", how="left", validate="one_to_one")
    enriched = enriched.merge(factor_state, on="score_day", how="left", validate="one_to_one")
    factor_state.to_csv(run_root / "factor_state_v1.csv", index=False)
    factor_audit.to_csv(run_root / "factor_input_audit.csv", index=False)
    _write_json(run_root / "selector_input_verification.json", selector_verification)
    _write_json(run_root / "factor_input_verification.json", factor_verification)

    families = _families(path_features, geometry_features, factor_features)
    _write_json(
        run_root / "preregistered_state_families.json",
        [
            {**{key: value for key, value in family.items() if key != "blocks"}, "blocks": {name: list(columns) for name, columns in family["blocks"].items()}}
            for family in families
        ],
    )
    all_daily: list[pd.DataFrame] = []
    for family in families:
        daily = run_block_basegap_variant(enriched, blocks=family["blocks"], variant=str(family["variant"]))
        daily["family"] = family["family"]
        daily["is_stage_a_reference"] = bool(family["is_stage_a_reference"])
        all_daily.append(daily)
    combined = pd.concat(all_daily, ignore_index=True)

    stage_a_daily = pd.read_csv(stage_a_run / "daily_selector_replay.csv")
    expected_path = stage_a_daily.loc[stage_a_daily["variant"] == "lb240_k60_trim20_lcb10"].copy()
    actual_path = combined.loc[combined["variant"] == "path44_only"].copy()
    path_parity = parity_audit(
        actual_path[
            ["score_day", "basegap_selected_name", "basegap_reason", "basegap_history_days", "basegap_history_end", "basegap_score_gap", *(f"basegap_score_{model}" for model in MODELS)]
        ],
        expected_path[
            ["score_day", "basegap_selected_name", "basegap_reason", "basegap_history_days", "basegap_history_end", "basegap_score_gap", *(f"basegap_score_{model}" for model in MODELS)]
        ],
        label="path44_block_distance_vs_stage_a",
    )
    path_parity.to_csv(run_root / "path44_stage_a_parity_audit.csv", index=False)
    combined.to_csv(run_root / "daily_state_family_replay.csv", index=False)
    summary = _family_summary(combined, families)
    summary.to_csv(run_root / "summary_metrics.csv", index=False)
    ranking = rank_design_families(summary)
    ranking.to_csv(run_root / "design_ranking.csv", index=False)
    _plot_state_summary(run_root, summary)
    active = combined.loc[combined["selection_active"]]
    _write_json(
        run_root / "leakage_audit.json",
        {
            "strict_predecessor_neighbours": bool((pd.to_datetime(active["neighbor_max_day"]).dt.date < pd.to_datetime(active["score_day"]).dt.date).all()),
            "factor_temporal_normalisation": {"lookback": FACTOR_TEMPORAL_LOOKBACK, "min_periods": FACTOR_TEMPORAL_MIN_PERIODS, "history_rule": "strictly date < score_day"},
            "factor_score_universe_rule": "exact three-way common frozen score code set on score_day",
            "no_return_or_label_input_to_factor_state": True,
            "path44_stage_a_parity_verified": True,
            "final_oos_not_used_for_state_family_selection": True,
            "variants": int(combined["variant"].nunique()),
            "active_rows": int(len(active)),
        },
    )
    _write_results(run_root, source_run=source_run, ranking=ranking, summary=summary)
    winner = ranking.iloc[0]
    manifest = {
        "run_class": "research_only_pure_basegap_preregistered_state_families",
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "live_config_written": False,
        "live_result_written": False,
        "model_state_written": False,
        "selector_source_run": str(source_run),
        "stage_a_source_run": str(stage_a_run),
        "stage_a_daily_selector_sha256": sha256(stage_a_run / "daily_selector_replay.csv"),
        "selector_source_manifest_sha256": selector_verification["source_manifest_sha256"],
        "factor_freeze_manifest_sha256": factor_verification["factor_manifest_sha256"],
        "factor_contract": {"path": str(FACTOR_CONTRACT_PATH), "sha256": sha256(FACTOR_CONTRACT_PATH), "factor_count": len(factor_names)},
        "fixed_basegap_parameters": {"lookback_days": FIXED_LOOKBACK_DAYS, "nearest_k": FIXED_NEAREST_K, "min_periods": FIXED_MIN_PERIODS, "metric": FIXED_METRIC, "margin": 0.0},
        "factor_state_definition": {"name": "F12", "columns": list(factor_features), "legacy_factor_count": LEGACY_FACTOR_COUNT, "temporal_robust_standardisation": {"lookback": FACTOR_TEMPORAL_LOOKBACK, "min_periods": FACTOR_TEMPORAL_MIN_PERIODS}},
        "score_geometry_definition": {"name": "G19", "columns": list(geometry_features)},
        "state_families": [{**{key: value for key, value in family.items() if key != "blocks"}, "blocks": {name: list(columns) for name, columns in family["blocks"].items()}} for family in families],
        "design_selected_variant": str(winner["variant"]),
        "splits": [{"name": name, "start": start, "end": end} for name, start, end in SPLITS],
        "source_code": [{"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}],
        "git": {"head": _safe_git(["rev-parse", "HEAD"]), "short_head": _safe_git(["rev-parse", "--short", "HEAD"]), "status_porcelain": _safe_git(["status", "--porcelain"])},
        "completed_at_utc": datetime.now(timezone.utc),
    }
    _write_json(run_root / "run_manifest.json", manifest)
    _write_json(run_root / "run_status.json", {"status": "completed", "completed_at_utc": datetime.now(timezone.utc), "design_selected_variant": str(winner["variant"]), "database_writes": False, "live_runtime_called": False, "scheduler_called": False, "research_output_only": True})
    (run_root / "task_state.md").write_text(_task_state(phase="completed", run_root=run_root, source_run=source_run), encoding="utf-8")
    return {"run_root": str(run_root), "design_selected_variant": str(winner["variant"]), "design_selected_alpha_bp_per_day": float(winner["selection_alpha_mean_bp"]), "verified_factor_files": int(factor_verification["verified_used_factor_files"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--stage-a-run", type=Path, default=DEFAULT_STAGE_A_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = _make_run_root(args.output_root, args.run_name)
    result = run_replay(source_run=args.source_run, stage_a_run=args.stage_a_run, run_root=run_root)
    print(json.dumps(result, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
