"""Frozen, research-only LightGBM three-candidate ranking replay.

This is deliberately a *model selector* experiment, not a new bond scorer.  A
score day has exactly three documents/query rows -- Regsim, Ensemble and HL20
-- and LambdaRank predicts their realised full-cycle return ordering.  The
selected action is a direct argmax with no confidence threshold, Champion
preference, Robust veto, BaseGap gate, database write, scheduler call or live
configuration mutation.

Inputs are deliberately bounded to the existing frozen live50 research
snapshot and the separately frozen live50 factor store:

* same-day three-candidate score geometry and candidate-specific score shape;
* candidate Top20 factor exposure and score-factor relationship summaries;
* causal, day-level 50-factor structure summaries;
* strictly earlier, execution-metadata-complete relative candidate returns.

The current path44 state is intentionally excluded.  Its historical 14:30
provenance is not a strict 14:29 certificate.  Industry is intentionally
excluded too: a point-in-time industry SCD/available-at contract is not yet
available.  Therefore this replay cannot be used to promote a live selector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:  # statsmodels is optional for the research result, not model fitting.
    import statsmodels.api as sm
except Exception:  # pragma: no cover - exercised only on reduced local environments.
    sm = None

from harness.tools.model_switch_basegap_state_family_replay import (
    DEFAULT_FACTOR_FREEZE_ROOT,
    causal_robust_z,
    daily_factor_statistics,
    load_factor_cross_section,
    verify_factor_freeze,
)
from harness.tools.model_switch_basegap_tuning_replay import (
    DEFAULT_SOURCE_RUN,
    MODELS,
    SPLITS,
    sha256,
    validate_frozen_source,
)


DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\model_switch_lgbm_candidate_ranker_20260814")
REPORT_START = "2024-05-08"
REPORT_END = "2026-07-30"
TOP_K = 20
TRAINING_WINDOW_DAYS = 360
MIN_TRAINING_DAYS = 120
RANDOM_SEED = 20260814
LEGACY_FACTOR_COUNT = 27
DECISION_TIME = time(hour=14, minute=29)
SELL_LABEL_AVAILABLE_TIME = time(hour=9, minute=39)
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
FACTOR_MISSING_POLICY = "fail_closed_for_this_replay: every requested feature must be finite for all three candidates"

# This is intentionally a literal, not a read from the dirty working-tree
# config.  It was frozen from the immutable live50 factor snapshot and its
# ordered schema is verified against that snapshot at every run.  The first 27
# names are the legacy live factors and the final 23 are the admitted mined
# factors; the split is part of the research contract, not a runtime lookup.
FROZEN_LIVE50_FACTOR_CONTRACT: tuple[str, ...] = (
    "cb_overnight_return_mean_20d",
    "cb_overnight_return_mean_5d",
    "cb_overnight_return_mean_60d",
    "range_30m",
    "cb_overnight_return_mean_10d",
    "amount_30m",
    "vol_30m",
    "alpha001_signed_power_v1",
    "alpha030_close_sign_volume_v1",
    "volume_30m",
    "depth_weighted_imbalance_v1",
    "alpha024_close_trend_filter_v1",
    "cb_overnight_sharpe_20_0930_0935",
    "daily_sharpe_twap_5d_mean5",
    "cb_overnight_sharpe_5_0930_0935",
    "mid_move_30m",
    "premium_momentum_proxy_v1",
    "volen_f60_s10_l3",
    "daily_sharpe_twap_20d_mean5",
    "cb_overnight_return_mean_40d",
    "alpha041_geometric_mean_vwap_v1",
    "alpha078_low_vwap_adv_corr_v1",
    "mom_slope_30m",
    "alpha019_close_momentum_sign_v1",
    "ret_10m",
    "alpha025_return_volume_vwap_range_v1",
    "alpha050_volume_vwap_corr_max_v1",
    "base_debt_premium_floor_gap",
    "bsfst_stock_return_bond_flow_mutual_information60",
    "bssrc_bond_stock_rank_correlation60",
    "bstk_tail_cocrash_residual20",
    "dliq_volume_return_corr20",
    "dohw_intraday_sign_range_asymmetry60",
    "dohw_mean_wick_asymmetry60",
    "dredemption_bondpremium_interaction",
    "dredemption_premium_z20",
    "dret_drawup_drawdown_asym",
    "dret_volatility_20",
    "drt_rebound_from_low20",
    "dtwap_morning_slope20",
    "lcc_amount_trade_size_information60",
    "lcc_volume_deal_information60",
    "lrd_cross_side_reprice_symmetry",
    "prcn_return_capacity_rank_corr60",
    "qed_prior_quote_lag2_agreement",
    "qed_prior_quote_location_dispersion",
    "qed_prior_quote_tail_penetration",
    "rjst_amount_joint_transition_entropy60",
    "rlmi_return_deal_sign_mutual_information60",
    "ydpt_yield_fall_return_beta60",
)


def _factor_contract_digest(factor_names: Sequence[str]) -> str:
    payload = {
        "contract_id": "frozen_live50_20260805_27plus23",
        "legacy_factor_count": LEGACY_FACTOR_COUNT,
        "factors": list(factor_names),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


FROZEN_LIVE50_FACTOR_CONTRACT_SHA256 = _factor_contract_digest(FROZEN_LIVE50_FACTOR_CONTRACT)
EXPECTED_FACTOR_FREEZE_MANIFEST_SHA256 = "9df6f0c08413642f28f8e7adb71d16221f1d086c5831b8a13252a446e221990c"
EXPECTED_FACTOR_FREEZE_TREE_SHA256 = "3eb30a6f9576b0f9887642ae879aec000e7d38d48423fb58c39789ab1e4b99b4"
EXPECTED_SELECTOR_SOURCE_MANIFEST_SHA256 = "23fcef9ec79ccbf6bccce01ba6c108540bf90b933853bfb3297ac3500e059d0f"
FACTOR_PIT_EVIDENCE_LEVEL = "historical_config_inference_not_frozen_artifact"
SCORE_PIT_EVIDENCE_LEVEL = "frozen_csv_hash_and_trade_date_only_no_available_at_certificate"
FACTOR_CONTRACT_SOURCE_EVIDENCE = {
    "source_git_commit": "009d831d790a30573be3301d9b9b905b1d919387",
    "live50_contract_sha256": "9d478e17bf07daca4318443774064fc422df79dc1202db321ecc09b04174adc4",
    "legacy27_pack_sha256": "1862929dfee680b0810964152b490ec94e30b9e0ff8e1397ad0331af7df3ad32",
    "mined23_profile_sha256": "5b646b3c44c1df5021c593b744eb9d02f7b37967d0a6c451dba61a699d8a5e86",
    "rust50_specs_sha256": "458af9bcfeb58ab4b344f4daec9d637b7a599fb572574c5bca142b30ce8d15b0",
    "panel_config_sha256": "83a5c43c2c1265c3da1a49e336a42898e786f4024554f0a13bf331666ecc6a94",
    "temporal_inference": "panel_config T1430 with lead_minutes=1 and panel.py end-minus-lead semantics implies <=14:29, but the frozen parquet bytes carry no configuration provenance",
}


def _assert_expected_source_manifest(source_verification: Mapping[str, object]) -> None:
    actual = str(source_verification.get("source_manifest_sha256", ""))
    if actual != EXPECTED_SELECTOR_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("frozen selector source manifest SHA-256 drifted")

# This is intentionally one fixed, low-capacity parameter point.  It is not a
# hyperparameter grid: with only a few hundred independent day groups, a broad
# tree search would simply recreate the overfit problem under investigation.
LGBM_PARAMS: dict[str, object] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 60,
    "learning_rate": 0.05,
    "num_leaves": 3,
    "max_depth": 2,
    "min_child_samples": 60,
    "min_split_gain": 0.05,
    "reg_alpha": 1.0,
    "reg_lambda": 20.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

CANDIDATE_ID_COLUMNS = tuple(f"candidate_is_{model}" for model in MODELS)
GEOMETRY_FEATURES = (
    *CANDIDATE_ID_COLUMNS,
    "score_top20_mean_z",
    "score_top20_gap_z",
    "score_top20_std_z",
    "score_skew",
    "score_tail_asymmetry",
    "score_pair_spearman_mean",
    "score_pair_jaccard_mean",
    "score_pair_rank_absdiff_mean",
    "score_top20_union_intersection",
    "score_prev_top20_jaccard",
    "lag_relative_ewm5",
    "lag_relative_ewm20",
    "lag_relative_ewm60",
)
FACTOR_EXPOSURE_FEATURES = (
    "factor_top20_rank_mean_legacy",
    "factor_top20_rank_iqr_legacy",
    "factor_top20_missing_rate_legacy",
    "factor_top20_rank_dispersion_legacy",
    "factor_score_rho_mean_legacy",
    "factor_score_absrho_mean_legacy",
    "factor_top20_rank_mean_mined",
    "factor_top20_rank_iqr_mined",
    "factor_top20_missing_rate_mined",
    "factor_top20_rank_dispersion_mined",
    "factor_score_rho_mean_mined",
    "factor_score_absrho_mean_mined",
)
FACTOR_STATE_FEATURES = (
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
)
VARIANTS: dict[str, tuple[str, ...]] = {
    "lgbm_geometry_lag": GEOMETRY_FEATURES,
    "lgbm_geometry_factor_exposure": (*GEOMETRY_FEATURES, *FACTOR_EXPOSURE_FEATURES, *FACTOR_STATE_FEATURES),
}


@dataclass(frozen=True)
class ReplayResult:
    daily: pd.DataFrame
    importance: pd.DataFrame


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


def _make_run_root(output_root: Path, run_name: str | None) -> Path:
    root = _resolve_inside(output_root, DEFAULT_OUTPUT_ROOT, description="research output root")
    name = run_name or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"run name must be one plain path component, got {name!r}")
    run_root = _resolve_inside(root / name, DEFAULT_OUTPUT_ROOT, description="research run")
    if run_root == DEFAULT_OUTPUT_ROOT.resolve() or run_root.exists():
        raise FileExistsError(f"refusing to overwrite research output: {run_root}")
    return run_root


def _factor_slice_path(factor_root: Path, score_day: date) -> Path:
    return factor_root / "factors" / "T1430" / f"{score_day:%Y-%m}" / f"{score_day:%Y%m%d}.parquet"


def _verify_frozen_factor_contract(factor_root: Path, score_days: Sequence[object]) -> dict[str, object]:
    """Fail closed unless the immutable factor schema matches the literal contract."""

    factor_names = FROZEN_LIVE50_FACTOR_CONTRACT
    if len(factor_names) != 50 or len(set(factor_names)) != 50:
        raise AssertionError("frozen live50 factor contract is malformed")
    if _factor_contract_digest(factor_names) != FROZEN_LIVE50_FACTOR_CONTRACT_SHA256:
        raise AssertionError("frozen live50 factor contract digest drifted")
    days = sorted({pd.Timestamp(day).date() for day in score_days})
    if not days:
        raise ValueError("cannot validate an empty factor score-day set")
    observed_times: set[str] = set()
    schema_hashes: list[str] = []
    for day in days:
        path = _factor_slice_path(factor_root, day)
        frame = pd.read_parquet(path)
        if tuple(frame.columns) != factor_names:
            raise ValueError(f"frozen factor physical schema/order drift: {path}")
        if not isinstance(frame.index, pd.MultiIndex) or tuple(frame.index.names) != ("dt", "code"):
            raise ValueError(f"frozen factor schema must use (dt, code) index: {path}")
        timestamps = pd.to_datetime(frame.index.get_level_values("dt"), errors="coerce")
        if timestamps.isna().any() or set(timestamps.date) != {day}:
            raise ValueError(f"frozen factor date drift: {path}")
        observed_times.update(value.strftime("%H:%M:%S") for value in timestamps)
        if not frame.notna().any(axis=0).all():
            raise ValueError(f"frozen factor has an all-missing contract column: {path}")
        schema_hashes.append(sha256(path))
    if sorted(observed_times) != ["14:30:00"]:
        raise ValueError(f"expected physical T1430 factor slices, got {sorted(observed_times)}")
    return {
        "contract_id": "frozen_live50_20260805_27plus23",
        "factor_count": len(factor_names),
        "legacy_factor_count": LEGACY_FACTOR_COUNT,
        "factors": list(factor_names),
        "cohorts": {"legacy": list(factor_names[:LEGACY_FACTOR_COUNT]), "mined": list(factor_names[LEGACY_FACTOR_COUNT:])},
        "sha256": FROZEN_LIVE50_FACTOR_CONTRACT_SHA256,
        "verified_full_schema_days": len(days),
        "verified_factor_file_sha256_digest": hashlib.sha256("".join(schema_hashes).encode("ascii")).hexdigest(),
        "physical_slice_timestamp": "14:30:00",
        "pit_evidence_level": FACTOR_PIT_EVIDENCE_LEVEL,
        "pit_certificate": "frozen T1430 input; this replay relies on the historical configuration inference below and does not independently certify the 14:29 cutoff",
        "source_contract_evidence": FACTOR_CONTRACT_SOURCE_EVIDENCE,
    }


def _score_path(input_root: Path, model: str, score_day: date) -> Path:
    return input_root / "scores" / model / f"{score_day:%Y-%m}" / f"{score_day:%Y-%m-%d}.csv"


def _read_aligned_scores(input_root: Path, score_day: date) -> pd.DataFrame:
    """Load the three frozen score vectors on exactly one common code universe."""

    codes: pd.Index | None = None
    expected: set[str] | None = None
    score_vectors: dict[str, pd.Series] = {}
    for model in MODELS:
        path = _score_path(input_root, model, score_day)
        header = pd.read_csv(path, nrows=0).columns.tolist()
        required = {"trade_date", "code", "score"}
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"frozen score CSV lacks {missing}: {path}")
        frame = pd.read_csv(path, usecols=["trade_date", "code", "score"])
        trade_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        if trade_dates.isna().any() or not trade_dates.eq(score_day).all():
            raise ValueError(f"frozen score CSV trade_date does not match path score_day={score_day}: {path}")
        frame["code"] = frame["code"].astype(str)
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        frame = frame.dropna(subset=["code", "score"])
        if frame.empty or frame["code"].duplicated().any():
            raise ValueError(f"invalid frozen score universe (empty or duplicate code): {model} {score_day}")
        model_codes = pd.Index(frame["code"].tolist(), name="code")
        model_set = set(model_codes)
        if expected is None:
            expected = model_set
            codes = model_codes
        elif model_set != expected:
            raise ValueError(f"score universe changed after common-code audit: {model} {score_day}")
        score_vectors[model] = frame.set_index("code")["score"]
    if codes is None:
        raise AssertionError("no candidate score vector loaded")
    result = pd.DataFrame(index=codes)
    for model in MODELS:
        result[model] = score_vectors[model].reindex(codes).to_numpy(dtype=float)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite aligned candidate score on {score_day}")
    return result


def _decision_timestamp(score_day: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(score_day, DECISION_TIME, tzinfo=SHANGHAI_TZ))


def _label_available_timestamp(sell_day: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(sell_day, SELL_LABEL_AVAILABLE_TIME, tzinfo=SHANGHAI_TZ))


def read_return_panel_with_maturity(return_paths: Mapping[str, Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load frozen returns while retaining the execution/maturity contract.

    A score-day label is eligible only after its next-day morning sell is
    complete.  The input snapshot records dates but not an exchange timestamp,
    so 09:39 Asia/Shanghai is an explicit frozen execution-contract assumption
    and is audited against every rolling decision at 14:29.
    """

    panel: pd.DataFrame | None = None
    metadata = ("score_day", "signal_day", "buy_day", "sell_day")
    required = {"trade_date", "day_return", *metadata}
    for model in MODELS:
        path = return_paths[model]
        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"frozen return file lacks execution-maturity columns {missing}: {path}")
        frame = pd.read_csv(path, usecols=["trade_date", "day_return", *metadata])
        parsed = {column: pd.to_datetime(frame[column], errors="coerce").dt.date for column in ("trade_date", *metadata)}
        if parsed["trade_date"].isna().any():
            raise ValueError(f"frozen return file has invalid trade_date: {path}")
        scored_rows = parsed["score_day"].notna()
        if scored_rows.any() and not parsed["trade_date"].loc[scored_rows].eq(parsed["score_day"].loc[scored_rows]).all():
            raise ValueError(f"frozen return trade_date/score_day mismatch: {path}")
        complete = pd.DataFrame({column: parsed[column] for column in metadata}).notna().all(axis=1)
        if complete.any():
            complete_rows = frame.index[complete]
            if not parsed["signal_day"].loc[complete_rows].eq(parsed["score_day"].loc[complete_rows]).all():
                raise ValueError(f"frozen return signal_day mismatch: {path}")
            if not parsed["buy_day"].loc[complete_rows].eq(parsed["score_day"].loc[complete_rows]).all():
                raise ValueError(f"frozen return buy_day is not score_day: {path}")
            if not (parsed["sell_day"].loc[complete_rows] > parsed["buy_day"].loc[complete_rows]).all():
                raise ValueError(f"frozen return sell_day is not after buy_day: {path}")
        frame["score_day"] = parsed["score_day"]
        frame[model] = pd.to_numeric(frame["day_return"], errors="coerce")
        frame[f"execution_metadata_complete_{model}"] = complete.astype(bool)
        frame[f"buy_day_{model}"] = parsed["buy_day"]
        frame[f"sell_day_{model}"] = parsed["sell_day"]
        frame[f"label_available_at_{model}"] = [
            _label_available_timestamp(day) if isinstance(day, date) else pd.NaT for day in parsed["sell_day"]
        ]
        frame = frame[[
            "score_day",
            model,
            f"execution_metadata_complete_{model}",
            f"buy_day_{model}",
            f"sell_day_{model}",
            f"label_available_at_{model}",
        ]]
        frame = frame.dropna(subset=["score_day", model]).drop_duplicates("score_day", keep="last")
        if frame.empty:
            raise ValueError(f"no usable return history for {model}: {path}")
        panel = frame if panel is None else panel.merge(frame, on="score_day", how="inner", validate="one_to_one")
    if panel is None or panel.empty:
        raise ValueError("no aligned candidate return history")
    completeness = [f"execution_metadata_complete_{model}" for model in MODELS]
    if not panel[completeness].nunique(axis=1).eq(1).all():
        raise ValueError("candidate return histories disagree on execution metadata completeness")
    panel["execution_metadata_complete"] = panel[completeness].all(axis=1)
    complete_rows = panel.loc[panel["execution_metadata_complete"]].copy()
    for field in ("buy_day", "sell_day", "label_available_at"):
        columns = [f"{field}_{model}" for model in MODELS]
        if not complete_rows[columns].nunique(axis=1).eq(1).all():
            raise ValueError(f"candidate return histories disagree on {field}")
        panel[field] = panel[columns[0]]
    return panel.sort_values("score_day").reset_index(drop=True), {
        "label_source": "aligned standalone full-cycle day_return",
        "buy_day_contract": "score_day",
        "sell_day_contract": "strictly after buy_day",
        "label_available_at": "sell_day 09:39:00 Asia/Shanghai",
        "decision_at": "score_day 14:29:00 Asia/Shanghai",
        "maturity_rule": "a training label must be execution-metadata-complete and label_available_at < current decision_at",
    }


def _ordered_top_codes(series: pd.Series, *, top_k: int = TOP_K) -> list[str]:
    if len(series) <= top_k:
        raise ValueError(f"score universe has {len(series)} rows, needs > Top{top_k}")
    frame = pd.DataFrame({"code": series.index.astype(str), "score": series.to_numpy(dtype=float)})
    return frame.sort_values(["score", "code"], ascending=[False, True], kind="stable").head(top_k)["code"].tolist()


def _safe_mean(values: np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def _safe_iqr(values: np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)) if len(finite) else float("nan")


def _safe_std(values: np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.std(ddof=0)) if len(finite) else float("nan")


def _score_z(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=float)
    center = float(np.mean(values))
    scale = float(np.std(values, ddof=0))
    if not math.isfinite(scale) or scale <= 1e-12:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - center) / scale


def _score_shape_features(series: pd.Series, top_codes: Sequence[str]) -> dict[str, float]:
    z = _score_z(series)
    top = z.reindex(list(top_codes)).to_numpy(dtype=float)
    ordered = pd.DataFrame({"code": series.index.astype(str), "score": z.to_numpy(dtype=float)}).sort_values(
        ["score", "code"], ascending=[False, True], kind="stable"
    )
    gap = float(ordered.iloc[TOP_K - 1]["score"] - ordered.iloc[TOP_K]["score"])
    q10, q50, q90 = np.quantile(z.to_numpy(dtype=float), [0.10, 0.50, 0.90])
    denominator = float(q90 - q10)
    tail_asymmetry = float(((q90 - q50) - (q50 - q10)) / denominator) if abs(denominator) > 1e-12 else 0.0
    return {
        "score_top20_mean_z": _safe_mean(top),
        "score_top20_gap_z": gap,
        "score_top20_std_z": _safe_std(top),
        "score_skew": float(pd.Series(z).skew()),
        "score_tail_asymmetry": tail_asymmetry,
    }


def _pairwise_score_features(scores: pd.DataFrame, top_sets: Mapping[str, set[str]]) -> dict[str, dict[str, float]]:
    """Return candidate-specific summaries of its disagreement with the other two."""

    ranks = scores.rank(axis=0, method="average", pct=True)
    result: dict[str, dict[str, float]] = {}
    union = set().union(*top_sets.values())
    intersection = set.intersection(*top_sets.values())
    union_intersection = float(len(intersection) / len(union)) if union else float("nan")
    for model in MODELS:
        correlations: list[float] = []
        jaccards: list[float] = []
        rank_diffs: list[float] = []
        for other in MODELS:
            if other == model:
                continue
            correlation = scores[model].corr(scores[other], method="spearman")
            correlations.append(float(correlation))
            joined = top_sets[model] | top_sets[other]
            jaccards.append(float(len(top_sets[model] & top_sets[other]) / len(joined)) if joined else float("nan"))
            rank_diffs.append(float((ranks[model] - ranks[other]).abs().mean()))
        result[model] = {
            "score_pair_spearman_mean": _safe_mean(np.asarray(correlations, dtype=float)),
            "score_pair_jaccard_mean": _safe_mean(np.asarray(jaccards, dtype=float)),
            "score_pair_rank_absdiff_mean": _safe_mean(np.asarray(rank_diffs, dtype=float)),
            "score_top20_union_intersection": union_intersection,
        }
    return result


def _factor_block_summary(
    factor_ranks: pd.DataFrame,
    raw_factors: pd.DataFrame,
    score_rank: pd.Series,
    top_codes: Sequence[str],
    factor_names: Sequence[str],
    prefix: str,
) -> dict[str, float]:
    names = list(factor_names)
    ranked = factor_ranks[names]
    raw = raw_factors[names]
    top_ranked = ranked.reindex(list(top_codes))
    top_raw = raw.reindex(list(top_codes))
    correlations = [score_rank.corr(ranked[name], method="spearman") for name in names]
    top_dispersion = [top_ranked[name].std(ddof=0) for name in names]
    return {
        f"factor_top20_rank_mean_{prefix}": _safe_mean(top_ranked.to_numpy(dtype=float)),
        f"factor_top20_rank_iqr_{prefix}": _safe_iqr(top_ranked.to_numpy(dtype=float)),
        f"factor_top20_missing_rate_{prefix}": float(top_raw.isna().to_numpy().mean()),
        f"factor_top20_rank_dispersion_{prefix}": _safe_mean(np.asarray(top_dispersion, dtype=float)),
        f"factor_score_rho_mean_{prefix}": _safe_mean(np.asarray(correlations, dtype=float)),
        f"factor_score_absrho_mean_{prefix}": _safe_mean(np.abs(np.asarray(correlations, dtype=float))),
    }


def _candidate_factor_features(
    *,
    scores: pd.DataFrame,
    factors: pd.DataFrame,
    top_codes: Mapping[str, Sequence[str]],
    factor_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    factor_ranks = factors[list(factor_names)].rank(axis=0, method="average", pct=True)
    legacy = tuple(factor_names[:LEGACY_FACTOR_COUNT])
    mined = tuple(factor_names[LEGACY_FACTOR_COUNT:])
    if len(legacy) != LEGACY_FACTOR_COUNT or len(mined) != 23:
        raise ValueError("expected the ordered 27+23 live50 factor contract")
    result: dict[str, dict[str, float]] = {}
    for model in MODELS:
        score_rank = scores[model].rank(method="average", pct=True)
        result[model] = {
            **_factor_block_summary(factor_ranks, factors, score_rank, top_codes[model], legacy, "legacy"),
            **_factor_block_summary(factor_ranks, factors, score_rank, top_codes[model], mined, "mined"),
        }
    return result


def _build_global_factor_state(
    *,
    raw_rows: Sequence[dict[str, object]],
    median_rows: Sequence[pd.Series],
    log_iqr_rows: Sequence[pd.Series],
    factor_names: Sequence[str],
) -> pd.DataFrame:
    raw = pd.DataFrame(raw_rows).sort_values("score_day").reset_index(drop=True)
    medians = pd.DataFrame(median_rows).reindex(columns=list(factor_names)).reset_index(drop=True)
    log_iqr = pd.DataFrame(log_iqr_rows).reindex(columns=list(factor_names)).reset_index(drop=True)
    z_medians = causal_robust_z(medians, lookback=120, min_periods=60)
    z_iqr = causal_robust_z(log_iqr, lookback=120, min_periods=60)
    raw["factor_loc_legacy"] = z_medians.iloc[:, :LEGACY_FACTOR_COUNT].median(axis=1, skipna=True)
    raw["factor_loc_mined"] = z_medians.iloc[:, LEGACY_FACTOR_COUNT:].median(axis=1, skipna=True)
    raw["factor_scale_legacy"] = z_iqr.iloc[:, :LEGACY_FACTOR_COUNT].median(axis=1, skipna=True)
    raw["factor_scale_mined"] = z_iqr.iloc[:, LEGACY_FACTOR_COUNT:].median(axis=1, skipna=True)
    return raw[["score_day", *FACTOR_STATE_FEATURES]].copy()


def _lag_relative_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Build only strictly-predecessor relative-return inputs for every score day."""

    if "label_available_at" not in returns.columns:
        raise KeyError("lagged-return construction requires label_available_at")
    ordered = returns.sort_values("score_day").reset_index(drop=True).copy()
    result = pd.DataFrame({"score_day": ordered["score_day"]})
    relative = ordered[list(MODELS)].sub(ordered[list(MODELS)].mean(axis=1), axis=0)
    complete = ordered["execution_metadata_complete"].astype(bool).to_numpy()
    label_available_at = pd.to_datetime(ordered["label_available_at"], errors="coerce", utc=True)
    for model in MODELS:
        for halflife in (5, 20, 60):
            values: list[float] = []
            for position in range(len(ordered)):
                history = relative.loc[: position - 1, model] if position else pd.Series(dtype=float)
                decision_at = _decision_timestamp(pd.Timestamp(ordered.loc[position, "score_day"]).date()).tz_convert("UTC")
                mature = (label_available_at.iloc[:position] < decision_at).to_numpy(dtype=bool)
                valid_mask = complete[:position] & mature
                history_values = history.to_numpy(dtype=float)[valid_mask] if len(history) else np.asarray([], dtype=float)
                history_values = history_values[np.isfinite(history_values)]
                if not len(history_values):
                    values.append(float("nan"))
                    continue
                ages = np.arange(len(history_values) - 1, -1, -1, dtype=float)
                weights = np.power(0.5, ages / float(halflife))
                values.append(float(np.average(history_values, weights=weights)))
            result[f"lag_relative_ewm{halflife}_{model}"] = values
    return result


def _relevance_and_rank(values: Mapping[str, float], model: str) -> tuple[int, int]:
    selected = float(values[model])
    rank = 1 + sum(float(value) > selected for value in values.values())
    return 3 - int(rank), int(rank)


def build_candidate_feature_panel(
    *,
    input_root: Path,
    factor_root: Path,
    returns: pd.DataFrame,
    factor_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exactly three candidate rows per frozen score day.

    The only realised returns present in the result are labels/audit fields.
    Lagged return features are constructed separately and use strictly earlier,
    execution-metadata-complete observations.
    """

    ordered_returns = returns.copy()
    ordered_returns["score_day"] = pd.to_datetime(ordered_returns["score_day"]).dt.date
    ordered_returns = ordered_returns.loc[
        (pd.to_datetime(ordered_returns["score_day"]) >= pd.Timestamp(REPORT_START))
        & (pd.to_datetime(ordered_returns["score_day"]) <= pd.Timestamp(REPORT_END))
    ].sort_values("score_day").reset_index(drop=True)
    lags = _lag_relative_features(ordered_returns).set_index("score_day")

    rows: list[dict[str, object]] = []
    factor_audit_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    median_rows: list[pd.Series] = []
    log_iqr_rows: list[pd.Series] = []
    previous_top_sets: dict[str, set[str]] | None = None

    for _, return_row in ordered_returns.iterrows():
        score_day = pd.Timestamp(return_row["score_day"]).date()
        scores = _read_aligned_scores(input_root, score_day)
        top_codes = {model: _ordered_top_codes(scores[model]) for model in MODELS}
        top_sets = {model: set(codes) for model, codes in top_codes.items()}
        pair_features = _pairwise_score_features(scores, top_sets)
        factors = load_factor_cross_section(
            factor_root=factor_root,
            score_day=score_day,
            score_codes=scores.index.tolist(),
            factor_names=factor_names,
        )
        valid_factor_count = factors.notna().sum(axis=1)
        if not factors.notna().any(axis=0).all():
            raise ValueError(f"factor contract has an all-missing column after score alignment on {score_day}")
        if int(valid_factor_count.min()) < LEGACY_FACTOR_COUNT:
            raise ValueError(
                f"factor readiness failed on {score_day}: a score-universe code has fewer than "
                f"{LEGACY_FACTOR_COUNT} valid frozen factors"
            )
        factor_features = _candidate_factor_features(
            scores=scores,
            factors=factors,
            top_codes=top_codes,
            factor_names=factor_names,
        )
        global_stats, medians, log_iqr = daily_factor_statistics(factors, factor_names)
        global_rows.append({"score_day": score_day, **global_stats})
        median_rows.append(medians.rename(score_day))
        log_iqr_rows.append(log_iqr.rename(score_day))
        factor_audit_rows.append(
            {
                "score_day": score_day,
                "score_universe_n": int(len(scores)),
                "factor_rows_after_alignment": int(len(factors)),
                "factor_cell_missing_rate": float(factors.isna().to_numpy().mean()),
                "factor_cells_finite": int(np.isfinite(factors.to_numpy(dtype=float)).sum()),
                "factor_cells_total": int(factors.shape[0] * factors.shape[1]),
                "factor_columns_with_any_finite": int(factors.notna().any(axis=0).sum()),
                "factor_columns_all_finite": int(factors.notna().all(axis=0).sum()),
                "score_universe_valid_factor_count_min": int(valid_factor_count.min()),
                "score_universe_valid_factor_count_median": float(valid_factor_count.median()),
                "factor_file": str(factor_root / "factors" / "T1430" / f"{score_day:%Y-%m}" / f"{score_day:%Y%m%d}.parquet"),
            }
        )
        realised = {model: float(return_row[model]) for model in MODELS}
        for model in MODELS:
            relevance, actual_rank = _relevance_and_rank(realised, model)
            prior_jaccard = float("nan")
            if previous_top_sets is not None:
                union = top_sets[model] | previous_top_sets[model]
                prior_jaccard = float(len(top_sets[model] & previous_top_sets[model]) / len(union)) if union else float("nan")
            row: dict[str, object] = {
                "score_day": score_day,
                "candidate": model,
                "execution_metadata_complete": bool(return_row["execution_metadata_complete"]),
                "buy_day": return_row["buy_day"],
                "sell_day": return_row["sell_day"],
                "label_available_at": return_row["label_available_at"],
                "relevance": int(relevance),
                "actual_rank": int(actual_rank),
                "realized_return": realised[model],
                **{f"realized_return_{name}": value for name, value in realised.items()},
                **{column: float(model == name) for column, name in zip(CANDIDATE_ID_COLUMNS, MODELS)},
                **_score_shape_features(scores[model], top_codes[model]),
                **pair_features[model],
                "score_prev_top20_jaccard": prior_jaccard,
                **factor_features[model],
            }
            for halflife in (5, 20, 60):
                row[f"lag_relative_ewm{halflife}"] = float(lags.at[score_day, f"lag_relative_ewm{halflife}_{model}"])
            rows.append(row)
        previous_top_sets = top_sets

    global_state = _build_global_factor_state(
        raw_rows=global_rows,
        median_rows=median_rows,
        log_iqr_rows=log_iqr_rows,
        factor_names=factor_names,
    )
    panel = pd.DataFrame(rows).merge(global_state, on="score_day", how="left", validate="many_to_one")
    panel = panel.sort_values(["score_day", "candidate"], key=lambda series: series.map({name: index for index, name in enumerate(MODELS)}) if series.name == "candidate" else series, kind="stable").reset_index(drop=True)
    counts = panel.groupby("score_day", sort=True)["candidate"].agg(list)
    if not counts.map(lambda values: tuple(values) == MODELS).all():
        raise RuntimeError("candidate panel must have Regsim/Ensemble/HL20 exactly once per score day")
    factor_feature_columns = [*FACTOR_EXPOSURE_FEATURES, *FACTOR_STATE_FEATURES]
    feature_audit_rows: list[dict[str, object]] = []
    for score_day, day_rows in panel.groupby("score_day", sort=True):
        values = day_rows[factor_feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        finite = np.isfinite(values)
        feature_audit_rows.append(
            {
                "score_day": score_day,
                "factor_feature_required_cells": int(values.size),
                "factor_feature_finite_cells": int(finite.sum()),
                "factor_feature_missing_cells": int((~finite).sum()),
                "factor_feature_all_finite": bool(finite.all()),
            }
        )
    factor_audit = pd.DataFrame(factor_audit_rows).merge(
        pd.DataFrame(feature_audit_rows), on="score_day", how="left", validate="one_to_one"
    )
    return panel, factor_audit


def _ordered_day_rows(panel: pd.DataFrame, score_day: date) -> pd.DataFrame:
    result = panel.loc[pd.to_datetime(panel["score_day"]).dt.date == score_day].copy()
    order = {model: position for position, model in enumerate(MODELS)}
    return result.sort_values("candidate", key=lambda value: value.map(order), kind="stable").reset_index(drop=True)


def _direct_argmax(values: Sequence[float]) -> int:
    array = np.asarray(values, dtype=float)
    if array.shape != (len(MODELS),) or not np.isfinite(array).all():
        raise ValueError("expected one finite ranking score per candidate")
    maximum = float(array.max())
    return int(next(index for index, value in enumerate(array) if float(value) == maximum))


def _feature_finiteness(rows: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[bool, int, int]:
    values = rows[list(feature_columns)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    finite = np.isfinite(values)
    return bool(finite.all()), int(finite.sum()), int(values.size)


def _labels_mature_before(rows: pd.DataFrame, decision_at: pd.Timestamp) -> bool:
    if not bool(rows["execution_metadata_complete"].all()):
        return False
    available = pd.to_datetime(rows["label_available_at"], errors="coerce", utc=True)
    if available.isna().any():
        return False
    return bool((available < decision_at.tz_convert("UTC")).all())


def run_rolling_ranker(
    panel: pd.DataFrame,
    *,
    variant: str,
    feature_columns: Sequence[str],
    training_window_days: int = TRAINING_WINDOW_DAYS,
    min_training_days: int = MIN_TRAINING_DAYS,
    params: Mapping[str, object] | None = None,
) -> ReplayResult:
    """Fit a fresh, strictly-predecessor low-capacity ranker for each day."""

    if training_window_days <= 0 or min_training_days <= 0 or min_training_days > training_window_days:
        raise ValueError("invalid training window/minimum")
    features = list(feature_columns)
    required = {
        "score_day",
        "candidate",
        "execution_metadata_complete",
        "label_available_at",
        "relevance",
        "realized_return",
        *features,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(f"candidate panel missing required columns: {missing}")
    ordered = panel.copy()
    ordered["score_day"] = pd.to_datetime(ordered["score_day"]).dt.date
    days = sorted(pd.unique(ordered["score_day"]).tolist())
    order = {model: position for position, model in enumerate(MODELS)}
    records: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    fitted_params = dict(LGBM_PARAMS if params is None else params)
    for position, score_day in enumerate(days):
        current = _ordered_day_rows(ordered, score_day)
        if len(current) != len(MODELS) or tuple(current["candidate"]) != MODELS:
            raise ValueError(f"candidate group is not exactly ordered {MODELS} on {score_day}")
        decision_at = _decision_timestamp(score_day)
        mature_prior_days: list[date] = []
        for prior_day in days[:position]:
            prior_rows = _ordered_day_rows(ordered, prior_day)
            if not _labels_mature_before(prior_rows, decision_at):
                continue
            mature_prior_days.append(prior_day)
        mature_window_days = mature_prior_days[-training_window_days:]
        train_days: list[date] = []
        for prior_day in mature_window_days:
            prior_rows = _ordered_day_rows(ordered, prior_day)
            features_finite, _, _ = _feature_finiteness(prior_rows, features)
            if features_finite:
                train_days.append(prior_day)
        training = ordered.loc[ordered["score_day"].isin(train_days)].copy()
        training = training.sort_values(["score_day", "candidate"], key=lambda series: series.map(order) if series.name == "candidate" else series, kind="stable")
        training_groups = [len(MODELS)] * len(train_days)
        input_ready, current_feature_finite_cells, current_feature_total_cells = _feature_finiteness(current, features)
        ready = bool(input_ready and len(train_days) >= min_training_days)
        training_label_available_at_max = (
            pd.to_datetime(training["label_available_at"], errors="coerce", utc=True).max() if not training.empty else pd.NaT
        )
        record: dict[str, object] = {
            "score_day": score_day,
            "decision_at": decision_at,
            "variant": variant,
            "feature_count": len(features),
            "current_feature_finite_cells": current_feature_finite_cells,
            "current_feature_total_cells": current_feature_total_cells,
            "current_feature_missing_cells": current_feature_total_cells - current_feature_finite_cells,
            "mature_label_days": int(len(mature_prior_days)),
            "mature_window_days": int(len(mature_window_days)),
            "usable_training_days": int(len(train_days)),
            "training_days": int(len(train_days)),
            "training_rows": int(len(training)),
            "training_end": train_days[-1] if train_days else None,
            "training_label_available_at_max": training_label_available_at_max,
            "prediction_ready": ready,
            "selection_reason": "lgbm_lambdarank_argmax" if ready else "warmup_or_input_or_maturity_unavailable_fixed_regsim",
            "selected_name": "Regsim",
            **{f"realized_return_{model}": float(current.loc[current["candidate"] == model, "realized_return"].iloc[0]) for model in MODELS},
            "execution_metadata_complete": bool(current["execution_metadata_complete"].all()),
        }
        predictions = np.full(len(MODELS), np.nan, dtype=float)
        if ready:
            if not _labels_mature_before(training, decision_at):
                raise RuntimeError(f"label maturity leakage before {score_day}")
            if not _feature_finiteness(training, features)[0]:
                raise RuntimeError(f"non-finite training feature after readiness filter before {score_day}")
            model = lgb.LGBMRanker(**fitted_params)
            x_train = training[features].replace([np.inf, -np.inf], np.nan)
            y_train = training["relevance"].astype(int)
            model.fit(x_train, y_train, group=training_groups)
            predictions = model.predict(current[features].replace([np.inf, -np.inf], np.nan)).astype(float)
            selected_index = _direct_argmax(predictions)
            record["selected_name"] = MODELS[selected_index]
            gain = model.booster_.feature_importance(importance_type="gain")
            split = model.booster_.feature_importance(importance_type="split")
            for name, gain_value, split_value in zip(features, gain, split):
                importance_rows.append(
                    {
                        "score_day": score_day,
                        "variant": variant,
                        "feature": name,
                        "gain": float(gain_value),
                        "split": int(split_value),
                    }
                )
        selected_name = str(record["selected_name"])
        actual = {model: float(record[f"realized_return_{model}"]) for model in MODELS}
        selected_return = actual[selected_name]
        best_name = next(model for model in MODELS if actual[model] == max(actual.values()))
        selected_rank = 1 + sum(value > selected_return for value in actual.values())
        record.update(
            {
                **{f"prediction_{model}": float(predictions[index]) for index, model in enumerate(MODELS)},
                "selected_return": selected_return,
                "equal_weight_return": float(np.mean(list(actual.values()))),
                "best_name": best_name,
                "best_return": actual[best_name],
                "selection_alpha": selected_return - float(np.mean(list(actual.values()))),
                "regret": actual[best_name] - selected_return,
                "selected_rank": int(selected_rank),
            }
        )
        records.append(record)
    return ReplayResult(daily=pd.DataFrame(records), importance=pd.DataFrame(importance_rows))


def _nav_statistics(values: pd.Series) -> dict[str, float | None]:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if series.empty:
        return {"mean_bp": None, "cumulative_return": None, "annualized_return": None, "sharpe": None, "max_drawdown": None}
    nav = (1.0 + series).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    daily_std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    return {
        "mean_bp": float(series.mean() * 10_000.0),
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": float(nav.iloc[-1] ** (252.0 / len(series)) - 1.0),
        "sharpe": float(series.mean() / daily_std * math.sqrt(252.0)) if math.isfinite(daily_std) and daily_std > 0.0 else None,
        "max_drawdown": float(drawdown.min()),
    }


def _paired_metrics(delta: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {"n_days": 0, "mean_bp": None, "iid_t": None, "iid_p": None, "hac_t": None, "hac_p": None}
    if len(values) > 1 and float(np.std(values, ddof=1)) > 0.0:
        test = stats.ttest_1samp(values, popmean=0.0)
        iid_t, iid_p = float(test.statistic), float(test.pvalue)
    else:
        iid_t, iid_p = None, None
    hac_t = hac_p = None
    if sm is not None and len(values) > 5:
        fit = sm.OLS(values, np.ones((len(values), 1), dtype=float)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        hac_t, hac_p = float(fit.tvalues[0]), float(fit.pvalues[0])
    return {
        "n_days": int(len(values)),
        "mean_bp": float(values.mean() * 10_000.0),
        "iid_t": iid_t,
        "iid_p": iid_p,
        "hac_t": hac_t,
        "hac_p": hac_p,
    }


def _scope_daily(daily: pd.DataFrame, *, start: str | None, end: str | None, ready_only: bool) -> pd.DataFrame:
    frame = daily.loc[daily["execution_metadata_complete"].astype(bool)].copy()
    if ready_only:
        frame = frame.loc[frame["prediction_ready"].astype(bool)].copy()
    if start is not None:
        frame = frame.loc[pd.to_datetime(frame["score_day"]) >= pd.Timestamp(start)].copy()
    if end is not None:
        frame = frame.loc[pd.to_datetime(frame["score_day"]) <= pd.Timestamp(end)].copy()
    return frame.sort_values("score_day").reset_index(drop=True)


def _metrics_row(daily: pd.DataFrame, *, variant: str, scope: str, ready_only: bool, start: str | None, end: str | None) -> dict[str, object]:
    frame = _scope_daily(daily, start=start, end=end, ready_only=ready_only)
    selected = _nav_statistics(frame["selected_return"])
    regsim = _nav_statistics(frame["realized_return_Regsim"])
    result: dict[str, object] = {
        "variant": variant,
        "scope": scope,
        "ready_only": bool(ready_only),
        "n_days": int(len(frame)),
        "start": None if frame.empty else str(pd.to_datetime(frame["score_day"]).min().date()),
        "end": None if frame.empty else str(pd.to_datetime(frame["score_day"]).max().date()),
        "prediction_ready_days": int(frame["prediction_ready"].sum()) if not frame.empty else 0,
        **{f"selected_{key}": value for key, value in selected.items()},
        **{f"Regsim_{key}": value for key, value in regsim.items()},
        "mean_rank": float(frame["selected_rank"].mean()) if not frame.empty else None,
        "rank1_rate": float((frame["selected_rank"] == 1).mean()) if not frame.empty else None,
        "rank3_rate": float((frame["selected_rank"] == 3).mean()) if not frame.empty else None,
        "mean_regret_bp": float(frame["regret"].mean() * 10_000.0) if not frame.empty else None,
        "model_switches": int(frame["selected_name"].ne(frame["selected_name"].shift()).sum() - 1) if len(frame) > 1 else 0,
    }
    result.update({f"vs_Regsim_{key}": value for key, value in _paired_metrics(frame["selected_return"] - frame["realized_return_Regsim"]).items()})
    return result


def build_summary_metrics(replays: Mapping[str, ReplayResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: tuple[tuple[str, str | None, str | None], ...] = (("full", REPORT_START, REPORT_END), *SPLITS)
    for variant, replay in replays.items():
        for scope, start, end in scopes:
            rows.append(_metrics_row(replay.daily, variant=variant, scope=scope, ready_only=True, start=start, end=end))
            rows.append(_metrics_row(replay.daily, variant=variant, scope=scope, ready_only=False, start=start, end=end))
    return pd.DataFrame(rows)


def _plot_nav(run_root: Path, replays: Mapping[str, ReplayResult]) -> None:
    figure, axis = plt.subplots(figsize=(13, 6))
    baseline_drawn = False
    for variant, replay in replays.items():
        daily = _scope_daily(replay.daily, start=REPORT_START, end=REPORT_END, ready_only=True)
        if daily.empty:
            continue
        axis.plot(pd.to_datetime(daily["score_day"]), (1.0 + daily["selected_return"].astype(float)).cumprod(), label=variant, linewidth=1.35)
        if not baseline_drawn:
            axis.plot(pd.to_datetime(daily["score_day"]), (1.0 + daily["realized_return_Regsim"].astype(float)).cumprod(), label="fixed Regsim", color="black", linewidth=2.0)
            baseline_drawn = True
    for _, _, boundary in SPLITS[:-1]:
        axis.axvline(pd.Timestamp(boundary), color="grey", linewidth=0.7, alpha=0.35)
    axis.set_title("Frozen LGBM candidate-ranker selector-shadow-return")
    axis.set_ylabel("Cumulative NAV")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(run_root / "nav_vs_regsim.png", dpi=160)
    plt.close(figure)


def _write_results(
    run_root: Path,
    summary: pd.DataFrame,
    importance: pd.DataFrame,
    *,
    training_window_days: int,
    min_training_days: int,
) -> None:
    ready_full = summary.loc[(summary["scope"] == "full") & summary["ready_only"]].copy()

    def _csv_block(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows."
        return "```csv\n" + frame.to_csv(index=False, float_format="%.6f").rstrip() + "\n```"

    lines = [
        "# Frozen LGBM three-candidate ranker replay",
        "",
        "## Contract",
        "",
        "- Research-only selector-shadow-return replay. No live config, live score, factor store, DB, scheduler, model state, or production result was written.",
        "- Query/group: one score day with exactly Regsim, Ensemble, and HL20; label relevance is realised standalone full-cycle net-return rank (2/1/0 with ties preserved).",
        "- Decision: direct LambdaRank argmax. There is no threshold, Champion role, Robust veto, BaseGap gate, or Fusion.",
        "- Inputs: frozen three-candidate score/factor files and strictly lagged complete returns only. Industry and legacy path44 state are excluded.",
        "- Factor slices are physically T1430 and score CSVs are frozen/hash-checked with trade-date checks. Neither input has an independently certified as-of/available-at proof; the factor <=14:29 claim is only historical configuration inference.",
        f"- Factor-input policy: {FACTOR_MISSING_POLICY}.",
        f"- Fixed training policy: fresh daily refit, max {training_window_days} strictly prior mature day groups, minimum {min_training_days}; fixed shallow LGBM parameters.",
        "- Primary baseline: fixed Regsim. Final OOS is report-only and never used to choose a variant or hyperparameter.",
        "",
        "## Full ready-window metrics",
        "",
        _csv_block(ready_full),
        "",
        "## Interpretation boundary",
        "",
        "- A better historical rank, NAV, or feature importance is not a live-promotion result. Require stable validation/final-OOS daily return advantage versus Regsim, risk control, and a separate forward shadow before any live consideration.",
        "- The factor-exposure variant is an ablation, not an ex-post selected deployment candidate.",
        "",
        "## Files",
        "",
        "- `candidate_feature_panel.csv`: three rows per score day with labels, label-maturity fields, and frozen factor/score features.",
        "- `daily_selector_replay.csv`: each daily direct choice and its realised counterfactual return.",
        "- `summary_metrics.csv`, `paired_vs_regsim.csv`, `feature_importance.csv`, `factor_input_audit.csv`, `frozen_input_manifest.json`, `leakage_audit.json`, `run_manifest.json`, and `nav_vs_regsim.png`.",
        "",
    ]
    if not importance.empty:
        top = importance.groupby(["variant", "feature"], as_index=False)[["gain", "split"]].mean().sort_values(["variant", "gain"], ascending=[True, False]).groupby("variant", as_index=False).head(10)
        lines.extend(["## Mean daily feature importance (diagnostic only)", "", _csv_block(top), ""])
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _task_state_text(*, phase: str, run_root: Path, detail: str) -> str:
    return f"""# Task State: LGBM three-candidate ranker replay

## Objective

- Frozen, research-only low-capacity LambdaRank selector for Regsim, Ensemble and HL20.

## Risk Level

- medium research risk; runtime and live assets are read-only.

## Current Facts

- Factor/score inputs are frozen; industry and path44 state are intentionally excluded.
- Factor slices are physical T1430; the claimed <=14:29 input cutoff remains historical-config inference rather than an independently frozen PIT certificate.
- Every training label retains its sell-day availability timestamp and must be complete before the current 14:29 decision.

## Artifacts

- output: `{run_root}`

## Phase

- {phase}: {detail}
"""


def run_replay(
    *,
    source_run: Path,
    output_root: Path,
    run_name: str | None = None,
    training_window_days: int = TRAINING_WINDOW_DAYS,
    min_training_days: int = MIN_TRAINING_DAYS,
) -> Path:
    """Verify frozen inputs, run the two fixed ablations, and return a new run root."""

    if training_window_days <= 0 or min_training_days <= 0 or min_training_days > training_window_days:
        raise ValueError("invalid training window/minimum")
    run_root = _make_run_root(output_root, run_name)
    run_root.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_root / "run_status.json",
        {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc),
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "training_window_days": training_window_days,
            "min_training_days": min_training_days,
        },
    )
    (run_root / "task_state.md").write_text(_task_state_text(phase="running", run_root=run_root, detail="validating frozen inputs"), encoding="utf-8")
    try:
        config, source_verification = validate_frozen_source(source_run)
        _assert_expected_source_manifest(source_verification)
        returns, label_maturity_contract = read_return_panel_with_maturity({
            model: Path(str(candidate["return_path"]["windows"])) if isinstance(candidate.get("return_path"), Mapping) else Path(str(candidate["return_path"]))
            for model, candidate in {
                "Regsim": config["champion"],
                "Ensemble": config["challengers"][0],
                "HL20": config["challengers"][1],
            }.items()
        })
        returns["score_day"] = pd.to_datetime(returns["score_day"]).dt.date
        returns = returns.loc[(pd.to_datetime(returns["score_day"]) >= pd.Timestamp(REPORT_START)) & (pd.to_datetime(returns["score_day"]) <= pd.Timestamp(REPORT_END))].copy()
        if returns.empty:
            raise RuntimeError("no returns inside locked report window")
        factor_names = FROZEN_LIVE50_FACTOR_CONTRACT
        factor_root, factor_verification = verify_factor_freeze(returns["score_day"].tolist())
        if factor_verification["factor_manifest_sha256"] != EXPECTED_FACTOR_FREEZE_MANIFEST_SHA256:
            raise RuntimeError("frozen factor manifest SHA-256 drifted")
        if factor_verification["factor_tree_sha256"] != EXPECTED_FACTOR_FREEZE_TREE_SHA256:
            raise RuntimeError("frozen factor tree SHA-256 drifted")
        factor_contract = _verify_frozen_factor_contract(factor_root, returns["score_day"].tolist())
        input_root = source_run.resolve() / "input_snapshot"
        frozen_input_manifest = {
            "source_run": str(source_run.resolve()),
            "source_verification": source_verification,
            "factor_verification": factor_verification,
            "factor_contract": factor_contract,
            "score_input_validation": "each candidate CSV must contain trade_date equal to its path score_day and the exact three-way common code universe",
            "score_pit_evidence_level": SCORE_PIT_EVIDENCE_LEVEL,
            "label_maturity_contract": label_maturity_contract,
            "rolling_training_policy": {
                "window_days": training_window_days,
                "min_training_days": min_training_days,
                "daily_refit": True,
                "warm_start": False,
            },
        }
        _write_json(run_root / "frozen_input_manifest.json", frozen_input_manifest)
        feature_panel, factor_audit = build_candidate_feature_panel(
            input_root=input_root,
            factor_root=factor_root,
            returns=returns,
            factor_names=factor_names,
        )
        replays = {
            variant: run_rolling_ranker(
                feature_panel,
                variant=variant,
                feature_columns=features,
                training_window_days=training_window_days,
                min_training_days=min_training_days,
            )
            for variant, features in VARIANTS.items()
        }
        daily = pd.concat([replay.daily for replay in replays.values()], ignore_index=True)
        importance = pd.concat([replay.importance for replay in replays.values()], ignore_index=True)
        summary = build_summary_metrics(replays)
        paired = summary[[column for column in summary.columns if column in {"variant", "scope", "ready_only", "n_days", "start", "end"} or column.startswith("vs_Regsim_")]].copy()
        feature_panel.to_csv(run_root / "candidate_feature_panel.csv", index=False)
        factor_audit.to_csv(run_root / "factor_input_audit.csv", index=False)
        daily.to_csv(run_root / "daily_selector_replay.csv", index=False)
        summary.to_csv(run_root / "summary_metrics.csv", index=False)
        paired.to_csv(run_root / "paired_vs_regsim.csv", index=False)
        importance.to_csv(run_root / "feature_importance.csv", index=False)
        _plot_nav(run_root, replays)
        ready_rows = daily.loc[daily["prediction_ready"].astype(bool)].copy()
        ready_label_maturity = bool(
            (
                pd.to_datetime(ready_rows["training_label_available_at_max"], errors="coerce", utc=True)
                < pd.to_datetime(ready_rows["decision_at"], errors="coerce", utc=True)
            ).all()
        ) if not ready_rows.empty else True
        leakage_audit = {
            "industry_features_included": False,
            "legacy_path44_state_included": False,
            "factor_contract": factor_contract,
            "factor_pit_status": "physical T1430 slice verified; <=14:29 provenance is historical-config inference, not an independently certified replay fact",
            "score_pit_status": "CSV hash and trade-date equality verified, but no independently certified as-of/available-at evidence exists",
            "score_universe_rule": "each score CSV trade_date equals its path score_day; exact frozen three-score common-code universe per day",
            "labels": "aligned standalone full-cycle day_return",
            "label_maturity_contract": label_maturity_contract,
            "training_uses_only_complete_labels": True,
            "training_labels_available_before_decision": ready_label_maturity,
            "strict_predecessor_training_end": bool(
                (pd.to_datetime(ready_rows["training_end"]).dt.date < pd.to_datetime(ready_rows["score_day"]).dt.date).all()
            ) if not ready_rows.empty else True,
            "factor_missing_policy": FACTOR_MISSING_POLICY,
            "direct_argmax_no_threshold": True,
            "group_size": len(MODELS),
            "candidate_order": list(MODELS),
            "report_window": {"start": REPORT_START, "end": REPORT_END},
            "rolling_training_policy": {
                "window_days": training_window_days,
                "min_training_days": min_training_days,
                "daily_refit": True,
                "warm_start": False,
            },
            "final_oos_not_used_for_variant_or_parameter_selection": True,
        }
        _write_json(run_root / "leakage_audit.json", leakage_audit)
        manifest = {
            "run_class": "research_only_lgbm_three_candidate_ranker",
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "source_run": str(source_run.resolve()),
            "source_verification": source_verification,
            "expected_source_manifest_sha256": EXPECTED_SELECTOR_SOURCE_MANIFEST_SHA256,
            "factor_verification": factor_verification,
            "frozen_input_manifest": str((run_root / "frozen_input_manifest.json").resolve()),
            "frozen_input_manifest_sha256": sha256(run_root / "frozen_input_manifest.json"),
            "factor_contract": factor_contract,
            "models": list(MODELS),
            "candidate_group_size": len(MODELS),
            "label": "same-score-day three-candidate realised standalone full-cycle day_return rank: best=2, middle=1, worst=0, ties preserved",
            "training": {
                "window_days": training_window_days,
                "min_training_days": min_training_days,
                "daily_refit": True,
                "warm_start": False,
                "label_maturity": label_maturity_contract,
                "factor_missing_policy": FACTOR_MISSING_POLICY,
            },
            "lgbm_params": LGBM_PARAMS,
            "variants": {key: list(value) for key, value in VARIANTS.items()},
            "excluded_inputs": {"industry": "no PIT SCD/available_at history", "path44_state": "historical 14:30 provenance not strict-1429 certified"},
            "source_code": [
                {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
                {
                    "path": str(REPO_ROOT / "harness" / "tools" / "model_switch_basegap_tuning_replay.py"),
                    "sha256": sha256(REPO_ROOT / "harness" / "tools" / "model_switch_basegap_tuning_replay.py"),
                },
                {
                    "path": str(REPO_ROOT / "harness" / "tools" / "model_switch_basegap_state_family_replay.py"),
                    "sha256": sha256(REPO_ROOT / "harness" / "tools" / "model_switch_basegap_state_family_replay.py"),
                },
            ],
            "library_versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            "git": {"head": _safe_git(["rev-parse", "HEAD"]), "status_porcelain": _safe_git(["status", "--short"])},
            "created_at_utc": datetime.now(timezone.utc),
        }
        _write_json(run_root / "run_manifest.json", manifest)
        _write_results(
            run_root,
            summary,
            importance,
            training_window_days=training_window_days,
            min_training_days=min_training_days,
        )
        (run_root / "task_state.md").write_text(
            _task_state_text(
                phase="completed",
                run_root=run_root,
                detail=(
                    "two predeclared factor/score ablations completed; "
                    f"daily refit with {training_window_days}-day rolling window and {min_training_days}-day minimum"
                ),
            ),
            encoding="utf-8",
        )
        _write_json(
            run_root / "run_status.json",
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc),
                "database_writes": False,
                "live_runtime_called": False,
                "scheduler_called": False,
            },
        )
        return run_root
    except Exception as exc:
        (run_root / "task_state.md").write_text(_task_state_text(phase="failed", run_root=run_root, detail=f"{type(exc).__name__}: {exc}"), encoding="utf-8")
        _write_json(
            run_root / "run_status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "database_writes": False,
                "live_runtime_called": False,
                "scheduler_called": False,
            },
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--training-window-days", type=int, default=TRAINING_WINDOW_DAYS)
    parser.add_argument("--min-training-days", type=int, default=MIN_TRAINING_DAYS)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_root = run_replay(
        source_run=args.source_run,
        output_root=args.output_root,
        run_name=args.run_name,
        training_window_days=args.training_window_days,
        min_training_days=args.min_training_days,
    )
    print(run_root)


if __name__ == "__main__":
    main()
