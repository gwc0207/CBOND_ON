"""Research-only replay of distance-weighted BaseGap scoring.

The production Base selector first finds the nearest 40 state-similar days
from a causal 60-day candidate window, then scores every model with an
equal-weight ``trim20_lcb10`` statistic.  This tool freezes the active live50
inputs into ``D:/cbond_on/research_scratch`` and compares that exact Base with
one deliberately fixed alternative:

* the 60-day window, Top-40 set, state z-score, candidate models, 5bp margin,
  Champion safeguards and Ridge Robust branch are unchanged;
* only the 40 observations receive Gaussian distance weights;
* the kernel has no return-tuned bandwidth: its raw weight is one half at the
  median Top-40 distance;
* the weighted score uses 20% fractional-mass trimming, weighted winsorised
  standard deviation and Kish effective sample size.

The replay calls the real Fusion routing after locally substituting only the
Base decision object.  It never invokes live runtime, DB writes, scheduling,
factor construction, model scoring, or any ``results/live`` output.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cbond_on.core.config import load_config_file
from cbond_on.infra.live import model_switch as live_model_switch
from cbond_on.infra.live.model_switch import SwitchDecision, _scoreopt_trim20_lcb10


RESEARCH_ROOT = Path(r"D:\cbond_on\research_scratch")
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "basegap_similarity_weight_20260806"
DEFAULT_LIVE_CONFIG = REPO_ROOT / "cbond_on" / "config" / "live" / "live_config.json5"
DEFAULT_STRATEGY_CONFIG = REPO_ROOT / "cbond_on" / "config" / "strategies" / "strategy01" / "strategy01_config.json5"
DEFAULT_LIVE_DECISION = Path(r"D:\cbond_on\results\live\2026-08-07\model_switch_decision.json")
EPSILON = 1e-12


@dataclass(frozen=True)
class Candidate:
    """One candidate model in the active Base/Fusion ordering."""

    role: str
    model_id: str
    name: str
    source_path: Path
    snapshot_path: Path


@dataclass(frozen=True)
class Snapshot:
    """All replay inputs after they have been copied out of live roots."""

    run_root: Path
    input_root: Path
    switch_cfg: dict[str, Any]
    candidates: tuple[Candidate, ...]
    state_path: Path
    return_panel: pd.DataFrame
    state_days: tuple[date, ...]
    live_reference_path: Path | None
    input_manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else f"git_error:{result.stderr.strip()}"


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def _finite(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _assert_output_root(output_root: Path) -> Path:
    allowed = RESEARCH_ROOT.resolve()
    output = output_root.resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError(f"research output must stay under {allowed}; received {output}")
    return output


def _path_for_windows(value: object, *, label: str) -> Path:
    if isinstance(value, dict):
        raw = value.get("windows")
    else:
        raw = value
    if not raw:
        raise ValueError(f"{label} has no Windows path")
    resolved = Path(str(raw)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _groups(switch_cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    champion = dict(switch_cfg.get("champion", {}))
    champion_id = str(champion.get("model_id", "")).strip()
    if not champion_id:
        raise ValueError("model_switch champion has no model_id")
    raw_challengers = switch_cfg.get("challengers")
    if isinstance(raw_challengers, list) and raw_challengers:
        challengers = [dict(item) for item in raw_challengers if isinstance(item, dict)]
    else:
        fallback = switch_cfg.get("challenger")
        challengers = [dict(fallback)] if isinstance(fallback, dict) else []
    if not challengers:
        raise ValueError("model_switch has no challengers")
    if any(not str(group.get("model_id", "")).strip() for group in challengers):
        raise ValueError("a model_switch challenger has no model_id")
    return champion, challengers


def _return_source(group: dict[str, Any], *, label: str) -> Path:
    raw = group.get("return_path") or group.get("score_return_path")
    return _path_for_windows(raw, label=label)


def _copy_input(source: Path, destination: Path, *, label: str) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot input: {destination}")
    shutil.copy2(source, destination)
    source_hash = _sha256(source)
    copied_hash = _sha256(destination)
    if source_hash != copied_hash:
        raise RuntimeError(f"copied input hash mismatch for {label}: {source}")
    return {
        "label": label,
        "source": str(source),
        "snapshot": str(destination),
        "sha256": source_hash,
        "bytes": int(source.stat().st_size),
    }


def _read_return_panel(candidates: Iterable[Candidate]) -> pd.DataFrame:
    panel: pd.DataFrame | None = None
    for candidate in candidates:
        frame = pd.read_csv(candidate.snapshot_path, usecols=["trade_date", "day_return"])
        frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        frame[candidate.model_id] = pd.to_numeric(frame["day_return"], errors="coerce")
        frame = frame[["score_day", candidate.model_id]].dropna().sort_values("score_day")
        frame = frame.drop_duplicates("score_day", keep="last")
        panel = frame if panel is None else panel.merge(frame, on="score_day", how="inner", validate="one_to_one")
    if panel is None or panel.empty:
        raise RuntimeError("the candidate return histories have no common complete dates")
    return panel.sort_values("score_day").reset_index(drop=True)


def _snapshot_inputs(
    *,
    run_root: Path,
    live_config_path: Path,
    strategy_config_path: Path,
    live_decision_path: Path | None,
) -> Snapshot:
    """Copy exactly the current selector inputs into the isolated run root."""

    live_config_path = live_config_path.resolve()
    strategy_config_path = strategy_config_path.resolve()
    if not live_config_path.exists():
        raise FileNotFoundError(f"live config missing: {live_config_path}")
    if not strategy_config_path.exists():
        raise FileNotFoundError(f"strategy config missing: {strategy_config_path}")

    source_config = load_config_file(live_config_path)
    switch_cfg_raw = source_config.get("model_switch")
    if not isinstance(switch_cfg_raw, dict):
        raise ValueError("live config has no model_switch mapping")
    switch_cfg = copy.deepcopy(switch_cfg_raw)
    if str(switch_cfg.get("mode", "")).strip() != "scoreopt_t1430_fusion_gate":
        raise ValueError(f"expected scoreopt_t1430_fusion_gate, received {switch_cfg.get('mode')}")

    strategy_cfg = load_config_file(strategy_config_path)
    turnover_ratio = _finite(strategy_cfg.get("turnover_ratio"))
    if turnover_ratio is None or not math.isclose(turnover_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            "this selector replay needs strategy01 turnover_ratio=1.0 so standalone shadow returns are valid"
        )

    input_root = run_root / "input_snapshot"
    input_root.mkdir(parents=False, exist_ok=False)
    copied: list[dict[str, Any]] = []
    copied.append(_copy_input(live_config_path, input_root / "source_live_config.json5", label="live_config"))
    copied.append(_copy_input(strategy_config_path, input_root / "strategy01_config.json5", label="strategy01_config"))

    state_source = _path_for_windows(switch_cfg.get("state_feature_path"), label="state_feature_path")
    state_path = input_root / "t1430_market_state_features_path_full.csv"
    copied.append(_copy_input(state_source, state_path, label="state_feature_history"))
    switch_cfg["state_feature_path"] = str(state_path)

    champion, challengers = _groups(switch_cfg)
    candidate_specs: list[tuple[str, dict[str, Any]]] = [("champion", champion)] + [
        (f"challenger_{index}", group) for index, group in enumerate(challengers)
    ]
    candidates: list[Candidate] = []
    destination_by_model: dict[str, Path] = {}
    for ordinal, (role, group) in enumerate(candidate_specs):
        model_id = str(group["model_id"]).strip()
        source_path = _return_source(group, label=f"{role}.return_path")
        destination = input_root / f"return_{ordinal}_{model_id}.csv"
        copied.append(_copy_input(source_path, destination, label=f"{role}_return_history"))
        destination_by_model[model_id] = destination
        candidates.append(
            Candidate(
                role="champion" if role == "champion" else "challenger",
                model_id=model_id,
                name=str(group.get("name") or model_id).strip(),
                source_path=source_path,
                snapshot_path=destination,
            )
        )

    # Rewrite all Base-facing return paths to the immutable copies.  The
    # redundant ``challenger`` field is preserved for configs that use it.
    switch_cfg["champion"] = dict(switch_cfg.get("champion", {}))
    switch_cfg["champion"]["return_path"] = str(destination_by_model[candidates[0].model_id])
    rewritten_challengers: list[dict[str, Any]] = []
    for group in switch_cfg.get("challengers", []) or []:
        item = dict(group)
        model_id = str(item.get("model_id", "")).strip()
        item["return_path"] = str(destination_by_model[model_id])
        rewritten_challengers.append(item)
    switch_cfg["challengers"] = rewritten_challengers
    if isinstance(switch_cfg.get("challenger"), dict):
        fallback = dict(switch_cfg["challenger"])
        fallback_id = str(fallback.get("model_id", "")).strip()
        if fallback_id in destination_by_model:
            fallback["return_path"] = str(destination_by_model[fallback_id])
        switch_cfg["challenger"] = fallback

    reference_path: Path | None = None
    if live_decision_path is not None and live_decision_path.exists():
        reference_path = input_root / "live_model_switch_decision.json"
        copied.append(_copy_input(live_decision_path.resolve(), reference_path, label="live_model_switch_decision"))

    state_frame = pd.read_csv(state_path, usecols=["trade_date"])
    state_days = tuple(
        sorted(
            {
                converted
                for converted in (_as_date(value) for value in state_frame["trade_date"])
                if converted is not None
            }
        )
    )
    if not state_days:
        raise RuntimeError("state history has no valid dates")
    panel = _read_return_panel(candidates)
    if panel.empty:
        raise RuntimeError("return panel is empty after snapshot")

    manifest = {
        "run_kind": "research_only_distance_weighted_basegap_replay",
        "created_at": datetime.now().astimezone().isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_head": _safe_git(["rev-parse", "HEAD"]),
        "git_status_porcelain": _safe_git(["status", "--porcelain"]),
        "model_switch_source_sha256": _sha256(REPO_ROOT / "cbond_on" / "infra" / "live" / "model_switch.py"),
        "strategy_contract": {
            "strategy_config": str(strategy_config_path),
            "turnover_ratio": turnover_ratio,
            "selector_shadow_equivalence": "turnover_ratio=1.0; standalone full-cycle day_return is valid for a fixed selected model",
        },
        "input_files": copied,
        "candidate_order": [asdict(candidate) for candidate in candidates],
        "return_panel": {
            "rows": int(len(panel)),
            "start_score_day": str(panel["score_day"].min()),
            "end_score_day": str(panel["score_day"].max()),
        },
        "state_days": {
            "rows": int(len(state_days)),
            "start_score_day": str(state_days[0]),
            "end_score_day": str(state_days[-1]),
        },
        "known_limitations": [
            "State history has the existing path_full_t1430 provenance; this replay does not certify a strict 14:29 state cutoff.",
            "This is a selector replay over active shadow return histories, not a new rolling model-training backtest.",
            "No parameter is selected using the replay returns; one fixed distance rule is evaluated.",
        ],
    }
    return Snapshot(
        run_root=run_root,
        input_root=input_root,
        switch_cfg=switch_cfg,
        candidates=tuple(candidates),
        state_path=state_path,
        return_panel=panel,
        state_days=state_days,
        live_reference_path=reference_path,
        input_manifest=manifest,
    )


def distance_weights(distances: np.ndarray) -> tuple[np.ndarray, dict[str, float | str | bool | None]]:
    """Return causal Gaussian similarity weights with no return-tuned knob.

    For non-degenerate Top-K distances, ``w_raw=2**(-(d/median(d))**2)``.
    Thus a day at the median distance has half the raw weight of an exact
    state match.  Equal distances deliberately return strict equal weights.
    """

    values = np.asarray(distances, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("distances must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all() or np.any(values < -EPSILON):
        raise ValueError("distances must be finite and non-negative")
    values = np.maximum(values, 0.0)
    count = int(len(values))
    median = float(np.median(values))
    equal_distance = bool(np.allclose(values, values[0], rtol=0.0, atol=EPSILON))
    if equal_distance or median <= EPSILON:
        weights = np.full(count, 1.0 / float(count), dtype=float)
        bandwidth = None
        mode = "equal_distance_equal_weight"
    else:
        bandwidth = float(median / math.sqrt(2.0 * math.log(2.0)))
        raw = np.exp(-0.5 * np.square(values / bandwidth))
        raw_sum = float(raw.sum())
        if not math.isfinite(raw_sum) or raw_sum <= 0.0:
            raise RuntimeError("distance kernel produced invalid total weight")
        weights = raw / raw_sum
        mode = "gaussian_median_half"
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise RuntimeError("distance kernel produced invalid weights")
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise RuntimeError("distance weights do not sum to one")
    return weights, {
        "weight_mode": mode,
        "distance_min": float(values.min()),
        "distance_p25": float(np.quantile(values, 0.25)),
        "distance_median": median,
        "distance_p75": float(np.quantile(values, 0.75)),
        "distance_max": float(values.max()),
        "kernel_bandwidth": bandwidth,
        "effective_sample_size": float(1.0 / np.dot(weights, weights)),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "equal_distance_fallback": equal_distance or median <= EPSILON,
    }


def _weighted_midmass_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    sorted_weights = sorted_weights / float(sorted_weights.sum())
    mid_mass = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    return float(np.interp(quantile, mid_mass, sorted_values, left=sorted_values[0], right=sorted_values[-1]))


def _weighted_trim_mean(values: np.ndarray, weights: np.ndarray, proportion: float = 0.20) -> float:
    if not 0.0 <= proportion < 0.5:
        raise ValueError("trim proportion must be within [0, 0.5)")
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    sorted_weights = sorted_weights / float(sorted_weights.sum())
    cumulative_end = np.cumsum(sorted_weights)
    cumulative_start = cumulative_end - sorted_weights
    retained = np.clip(
        np.minimum(cumulative_end, 1.0 - proportion) - np.maximum(cumulative_start, proportion),
        0.0,
        None,
    )
    retained_mass = float(retained.sum())
    if retained_mass <= EPSILON:
        raise RuntimeError("weighted trim retained no probability mass")
    return float(np.dot(sorted_values, retained) / retained_mass)


def weighted_trim20_lcb10(values: np.ndarray, weights: np.ndarray) -> tuple[float, dict[str, float | bool]]:
    """Weighted analogue of production ``trim20_lcb10``.

    The uniform branch calls the production implementation directly.  This is
    deliberately stronger than numerical similarity: it makes the experiment
    baseline exactly compatible with the live statistic when weights collapse
    to equal mass.
    """

    returns = np.asarray(values, dtype=float)
    normalized = np.asarray(weights, dtype=float)
    if returns.ndim != 1 or normalized.ndim != 1 or len(returns) != len(normalized) or not len(returns):
        raise ValueError("values and weights must be non-empty same-length vectors")
    if not np.isfinite(returns).all() or not np.isfinite(normalized).all() or np.any(normalized < 0.0):
        raise ValueError("values must be finite and weights must be finite/non-negative")
    normalized = normalized / float(normalized.sum())
    count = len(returns)
    uniform = np.full(count, 1.0 / float(count), dtype=float)
    effective_n = float(1.0 / np.dot(normalized, normalized))
    if np.allclose(normalized, uniform, rtol=0.0, atol=1e-14):
        center = float(live_model_switch._trim_mean(pd.Series(returns), 0.20))
        clipped = np.clip(returns, np.quantile(returns, 0.10), np.quantile(returns, 0.90))
        std = float(np.std(clipped, ddof=1)) if count > 1 else 0.0
        return _scoreopt_trim20_lcb10(pd.Series(returns)), {
            "trim_center": center,
            "winsor_low": float(np.quantile(returns, 0.10)),
            "winsor_high": float(np.quantile(returns, 0.90)),
            "winsor_std": std,
            "effective_sample_size": effective_n,
            "uniform_compatibility": True,
        }

    center = _weighted_trim_mean(returns, normalized, 0.20)
    lower = _weighted_midmass_quantile(returns, normalized, 0.10)
    upper = _weighted_midmass_quantile(returns, normalized, 0.90)
    clipped = np.clip(returns, lower, upper)
    clipped_mean = float(np.dot(normalized, clipped))
    denominator = 1.0 - float(np.dot(normalized, normalized))
    variance = 0.0 if denominator <= EPSILON else float(np.dot(normalized, np.square(clipped - clipped_mean)) / denominator)
    std = math.sqrt(max(variance, 0.0))
    score = float(center - std / math.sqrt(effective_n))
    return score, {
        "trim_center": center,
        "winsor_low": lower,
        "winsor_high": upper,
        "winsor_std": std,
        "effective_sample_size": effective_n,
        "uniform_compatibility": False,
    }


def _candidate_values(decision: SwitchDecision) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, tuple[date, ...]]:
    details = [dict(item) for item in (decision.candidate_scores or [])]
    if not details or not decision.similar_days:
        raise ValueError("Base decision has no candidate scores or similar days to weight")
    model_ids = [str(detail.get("model_id", "")).strip() for detail in details]
    if not all(model_ids) or len(set(model_ids)) != len(model_ids):
        raise ValueError("Base decision candidate model ids are missing or duplicated")
    distances: list[float] = []
    rows: list[list[float]] = []
    neighbour_days: list[date] = []
    for similar in decision.similar_days:
        neighbour_day = _as_date(similar.get("trade_date"))
        if neighbour_day is None or neighbour_day >= decision.score_day:
            raise ValueError(
                f"non-causal Base neighbour for {decision.score_day}: {similar.get('trade_date')}"
            )
        distance = _finite(similar.get("distance"))
        if distance is None:
            raise ValueError(f"Base neighbour has no finite distance for {decision.score_day}")
        returns_by_id: dict[str, float] = {}
        for entry in similar.get("model_returns", []) or []:
            model_id = str(entry.get("model_id", "")).strip()
            returned = _finite(entry.get("day_return"))
            if model_id and returned is not None:
                returns_by_id[model_id] = returned
        if any(model_id not in returns_by_id for model_id in model_ids):
            raise ValueError(f"Base neighbour lacks complete candidate returns for {decision.score_day}")
        distances.append(distance)
        rows.append([returns_by_id[model_id] for model_id in model_ids])
        neighbour_days.append(neighbour_day)
    return details, np.asarray(distances, dtype=float), np.asarray(rows, dtype=float), tuple(neighbour_days)


def weighted_base_decision(decision: SwitchDecision) -> tuple[SwitchDecision, dict[str, Any]]:
    """Re-score one production Base decision, keeping its sampled days fixed."""

    if not decision.similar_days or not decision.candidate_scores:
        return decision, {
            "weighting_applied": False,
            "skip_reason": "base_has_no_complete_similar_day_sample",
            "score_day": decision.score_day,
        }
    details, distances, values, neighbour_days = _candidate_values(decision)
    weights, weight_audit = distance_weights(distances)
    weighted_scores: list[float] = []
    score_audits: list[dict[str, float | bool]] = []
    for column in range(values.shape[1]):
        score, score_audit = weighted_trim20_lcb10(values[:, column], weights)
        weighted_scores.append(score)
        score_audits.append(score_audit)
    if not np.isfinite(np.asarray(weighted_scores, dtype=float)).all():
        raise RuntimeError(f"weighted Base scores are not finite for {decision.score_day}")

    ranked = sorted(range(len(details)), key=lambda index: (-weighted_scores[index], index))
    best_index, second_index = ranked[0], ranked[1]
    best_score = float(weighted_scores[best_index])
    second_score = float(weighted_scores[second_index])
    score_gap = float(best_score - second_score)
    champion_index = next(
        (index for index, detail in enumerate(details) if str(detail.get("model_id")) == decision.champion_model_id),
        None,
    )
    if champion_index is None:
        raise RuntimeError(f"champion not found in Base candidate scores for {decision.score_day}")
    selected_index = best_index if score_gap > float(decision.threshold) else champion_index
    selected = details[selected_index]
    best_challenger_index = next(
        index for index in ranked if str(details[index].get("model_id")) != decision.champion_model_id
    )
    best_challenger = details[best_challenger_index]
    updated_details: list[dict[str, Any]] = []
    for index, detail in enumerate(details):
        updated = dict(detail)
        updated["score"] = float(weighted_scores[index])
        updated_details.append(updated)

    weighted = replace(
        decision,
        metric="trim20_lcb10_distance_weighted_median_half",
        selected_model_id=str(selected["model_id"]),
        selected_name=str(selected.get("name") or selected["model_id"]),
        challenger_model_id=str(best_challenger["model_id"]),
        challenger_name=str(best_challenger.get("name") or best_challenger["model_id"]),
        champion_score=float(weighted_scores[champion_index]),
        challenger_score=float(weighted_scores[best_challenger_index]),
        score_diff=score_gap,
        reason="score_best" if score_gap > float(decision.threshold) else "margin_default",
        candidate_scores=updated_details,
    )
    audit: dict[str, Any] = {
        "score_day": decision.score_day,
        "weighting_applied": True,
        "neighbour_count": int(len(neighbour_days)),
        "neighbour_day_min": min(neighbour_days),
        "neighbour_day_max": max(neighbour_days),
        **weight_audit,
        "candidate_scores": {
            str(detail["model_id"]): {
                "name": str(detail.get("name") or detail["model_id"]),
                "equal_weight_score": _finite(detail.get("score")),
                "weighted_score": float(weighted_scores[index]),
                **score_audits[index],
            }
            for index, detail in enumerate(details)
        },
        "weighted_base_top_model_id": str(details[best_index]["model_id"]),
        "weighted_base_selected_model_id": weighted.selected_model_id,
        "weighted_base_reason": weighted.reason,
        "weighted_base_gap": weighted.score_diff,
    }
    return weighted, audit


def _full_fusion_with_base(switch_cfg: dict[str, Any], base: SwitchDecision) -> SwitchDecision:
    """Use real production Fusion while replacing only its just-built Base."""

    def _fixed_base(_cfg: dict, *, score_day: date) -> SwitchDecision:
        if score_day != base.score_day:
            raise AssertionError(f"cached Base day mismatch: {score_day} != {base.score_day}")
        return base

    with patch.object(live_model_switch, "decide_scoreopt_t1430_dispersion", _fixed_base):
        return live_model_switch.decide_scoreopt_t1430_fusion_gate(switch_cfg, score_day=base.score_day)


def _candidate_score_map(decision: SwitchDecision) -> dict[str, float | None]:
    return {
        str(item.get("model_id", "")): _finite(item.get("score"))
        for item in (decision.candidate_scores or [])
    }


def _fusion_action(decision: SwitchDecision) -> str | None:
    fusion = decision.fusion if isinstance(decision.fusion, dict) else {}
    action = fusion.get("action")
    return None if action is None else str(action)


def _replay(snapshot: Snapshot) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    state_day_set = set(snapshot.state_days)
    outcome_days = set(snapshot.return_panel["score_day"])
    evaluation_days = sorted(state_day_set & outcome_days)
    audit_only_days = sorted(state_day_set - outcome_days)
    if not evaluation_days:
        raise RuntimeError("no days have both state and complete candidate returns")

    outcome_lookup = snapshot.return_panel.set_index("score_day")
    daily_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    original_base = live_model_switch.decide_scoreopt_t1430_dispersion
    for score_day in [*evaluation_days, *audit_only_days]:
        equal_base = original_base(snapshot.switch_cfg, score_day=score_day)
        weighted_base, weight_audit = weighted_base_decision(equal_base)
        equal_full = _full_fusion_with_base(snapshot.switch_cfg, equal_base)
        weighted_full = _full_fusion_with_base(snapshot.switch_cfg, weighted_base)
        outcome_available = score_day in outcome_lookup.index
        equal_return: float | None = None
        weighted_return: float | None = None
        if outcome_available:
            row = outcome_lookup.loc[score_day]
            equal_return = _finite(row.get(equal_full.selected_model_id))
            weighted_return = _finite(row.get(weighted_full.selected_model_id))
            if equal_return is None or weighted_return is None:
                raise RuntimeError(f"selected model return missing for {score_day}")

        equal_scores = _candidate_score_map(equal_base)
        weighted_scores = _candidate_score_map(weighted_base)
        record: dict[str, Any] = {
            "score_day": score_day,
            "outcome_available": outcome_available,
            "equal_base_selected_model_id": equal_base.selected_model_id,
            "equal_base_selected_name": equal_base.selected_name,
            "equal_base_reason": equal_base.reason,
            "equal_base_gap": equal_base.score_diff,
            "weighted_base_selected_model_id": weighted_base.selected_model_id,
            "weighted_base_selected_name": weighted_base.selected_name,
            "weighted_base_reason": weighted_base.reason,
            "weighted_base_gap": weighted_base.score_diff,
            "base_selected_changed": equal_base.selected_model_id != weighted_base.selected_model_id,
            "equal_full_selected_model_id": equal_full.selected_model_id,
            "equal_full_selected_name": equal_full.selected_name,
            "equal_full_reason": equal_full.reason,
            "equal_full_action": _fusion_action(equal_full),
            "weighted_full_selected_model_id": weighted_full.selected_model_id,
            "weighted_full_selected_name": weighted_full.selected_name,
            "weighted_full_reason": weighted_full.reason,
            "weighted_full_action": _fusion_action(weighted_full),
            "full_selected_changed": equal_full.selected_model_id != weighted_full.selected_model_id,
            "equal_full_return": equal_return,
            "weighted_full_return": weighted_return,
            "weighted_minus_equal_return": None
            if equal_return is None or weighted_return is None
            else float(weighted_return - equal_return),
        }
        for index, candidate in enumerate(snapshot.candidates):
            record[f"equal_base_score_{index}"] = equal_scores.get(candidate.model_id)
            record[f"weighted_base_score_{index}"] = weighted_scores.get(candidate.model_id)
        daily_rows.append(record)

        flat_audit = {
            key: value
            for key, value in weight_audit.items()
            if key not in {"candidate_scores"}
        }
        for index, candidate in enumerate(snapshot.candidates):
            candidate_audit = (weight_audit.get("candidate_scores") or {}).get(candidate.model_id, {})
            flat_audit[f"equal_score_{index}"] = candidate_audit.get("equal_weight_score")
            flat_audit[f"weighted_score_{index}"] = candidate_audit.get("weighted_score")
            flat_audit[f"trim_center_{index}"] = candidate_audit.get("trim_center")
            flat_audit[f"winsor_std_{index}"] = candidate_audit.get("winsor_std")
        weight_rows.append(flat_audit)

    daily = pd.DataFrame(daily_rows).sort_values("score_day").reset_index(drop=True)
    weights = pd.DataFrame(weight_rows).sort_values("score_day").reset_index(drop=True)
    replay_meta = {
        "evaluation_score_days": int(len(evaluation_days)),
        "evaluation_start": str(evaluation_days[0]),
        "evaluation_end": str(evaluation_days[-1]),
        "audit_only_score_days": [str(value) for value in audit_only_days],
        "causality_check": "every weighted neighbour was asserted strictly earlier than its score_day",
    }
    return daily, weights, replay_meta


def _selector_metrics(daily: pd.DataFrame, *, return_column: str, selected_column: str) -> dict[str, Any]:
    frame = daily[daily["outcome_available"]].dropna(subset=[return_column]).copy()
    returns = pd.to_numeric(frame[return_column], errors="coerce").dropna().to_numpy(dtype=float)
    selected = frame.loc[frame[return_column].notna(), selected_column].astype(str).to_numpy()
    if not len(returns):
        raise RuntimeError(f"no returns available for {return_column}")
    nav = np.cumprod(1.0 + returns)
    volatility = float(np.std(returns, ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else float("nan")
    sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(252.0)) if len(returns) > 1 and volatility > 0.0 else float("nan")
    return {
        "days": int(len(returns)),
        "start_score_day": str(frame["score_day"].iloc[0]),
        "end_score_day": str(frame["score_day"].iloc[-1]),
        "total_return": float(nav[-1] - 1.0),
        "annualized_return": float((nav[-1]) ** (252.0 / len(returns)) - 1.0),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
        "switch_count": int(np.count_nonzero(selected[1:] != selected[:-1])) if len(selected) > 1 else 0,
    }


def _paired_metrics(daily: pd.DataFrame, *, changed_only: bool) -> dict[str, Any]:
    frame = daily[daily["outcome_available"]].dropna(subset=["weighted_minus_equal_return"]).copy()
    if changed_only:
        frame = frame[frame["full_selected_changed"]]
    values = frame["weighted_minus_equal_return"].to_numpy(dtype=float)
    if not len(values):
        return {"days": 0, "changed_only": changed_only}
    wins = int(np.count_nonzero(values > 0.0))
    nonzero = values[values != 0.0]
    two_sided = float(stats.ttest_1samp(values, 0.0).pvalue) if len(values) > 1 else float("nan")
    return {
        "days": int(len(values)),
        "changed_only": changed_only,
        "wins": wins,
        "losses": int(np.count_nonzero(values < 0.0)),
        "ties": int(np.count_nonzero(values == 0.0)),
        "mean_delta_bp": float(np.mean(values) * 1e4),
        "median_delta_bp": float(np.median(values) * 1e4),
        "sum_delta_bp": float(np.sum(values) * 1e4),
        "paired_t_p_two_sided": two_sided,
        "sign_p_two_sided": float(stats.binomtest(wins, len(nonzero), 0.5).pvalue) if len(nonzero) else float("nan"),
    }


def _weight_summary(weights: pd.DataFrame) -> dict[str, Any]:
    applied = weights[weights.get("weighting_applied", False).fillna(False)].copy()
    if applied.empty:
        return {"applied_days": 0}
    columns = ["effective_sample_size", "max_weight", "min_weight", "distance_median", "kernel_bandwidth"]
    return {
        "applied_days": int(len(applied)),
        "equal_distance_fallback_days": int(applied["equal_distance_fallback"].fillna(False).sum()),
        "effective_sample_size": {
            key: float(applied["effective_sample_size"].quantile(value))
            for key, value in {"min": 0.0, "p25": 0.25, "median": 0.5, "p75": 0.75, "max": 1.0}.items()
        },
        "maximum_single_weight": {
            key: float(applied["max_weight"].quantile(value))
            for key, value in {"min": 0.0, "p25": 0.25, "median": 0.5, "p75": 0.75, "max": 1.0}.items()
        },
        "available_audit_columns": [column for column in columns if column in applied.columns],
    }


def _write_nav_plot(run_root: Path, daily: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = daily[daily["outcome_available"]].dropna(subset=["equal_full_return", "weighted_full_return"]).copy()
    frame["equal_nav"] = (1.0 + frame["equal_full_return"].astype(float)).cumprod()
    frame["weighted_nav"] = (1.0 + frame["weighted_full_return"].astype(float)).cumprod()
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(frame["score_day"], frame["equal_nav"], label="current equal-weight BaseGap", linewidth=1.3)
    axis.plot(frame["score_day"], frame["weighted_nav"], label="distance-weighted BaseGap", linewidth=1.3)
    axis.set_title("Research-only BaseGap distance-weight replay")
    axis.set_ylabel("cumulative NAV")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    path = run_root / "nav_compare.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _live_reference_check(snapshot: Snapshot, daily: pd.DataFrame) -> dict[str, Any]:
    if snapshot.live_reference_path is None:
        return {"available": False}
    reference = json.loads(snapshot.live_reference_path.read_text(encoding="utf-8"))
    score_day = _as_date(reference.get("score_day"))
    if score_day is None:
        return {"available": True, "reproduced": False, "reason": "reference_has_no_score_day"}
    rows = daily[daily["score_day"] == score_day]
    if rows.empty:
        return {"available": True, "reproduced": False, "reason": "reference_day_not_replayed", "score_day": str(score_day)}
    row = rows.iloc[0]
    selected_match = str(row["equal_full_selected_model_id"]) == str(reference.get("selected_model_id"))
    expected_gap = _finite(reference.get("score_diff"))
    actual_gap = _finite(row.get("equal_base_gap"))
    gap_match = (expected_gap is None and actual_gap is None) or (
        expected_gap is not None and actual_gap is not None and math.isclose(expected_gap, actual_gap, abs_tol=1e-12, rel_tol=0.0)
    )
    return {
        "available": True,
        "score_day": str(score_day),
        "reproduced": bool(selected_match and gap_match),
        "selected_model_id_match": selected_match,
        "score_gap_match": gap_match,
        "reference_selected_model_id": reference.get("selected_model_id"),
        "replayed_selected_model_id": row["equal_full_selected_model_id"],
        "reference_base_gap": expected_gap,
        "replayed_base_gap": actual_gap,
    }


def _summary_markdown(
    *,
    summary: pd.DataFrame,
    paired_all: dict[str, Any],
    paired_changed: dict[str, Any],
    replay_meta: dict[str, Any],
    weight_meta: dict[str, Any],
    reference: dict[str, Any],
) -> str:
    rows = ["# BaseGap distance-weighted research replay", "", "## Fixed contract", ""]
    rows.extend(
        [
            "- Baseline: active live50 Base, causal 60-day candidate window, nearest 40, equal-weight `trim20_lcb10`.",
            "- Variant: same Top-40 rows; `w_i ∝ 2^(-(d_i / median(d))^2)`, so the median-distance row has half the raw weight of an exact match.",
            "- The variant retains weighted 20% fractional trimming and winsorized standard error divided by `sqrt(Kish ESS)`.",
            "- Full routing is replayed through the unchanged production Champion-first, Champion-third veto, Ridge Robust and Fusion code.",
            "- No live config, DB, scheduler, factor/model state, score output or live artifact was changed.",
            "",
            "## Evaluation window",
            "",
            f"- {replay_meta['evaluation_start']} through {replay_meta['evaluation_end']}; {replay_meta['evaluation_score_days']} realised score days.",
            f"- Audit-only days without realised returns: {', '.join(replay_meta['audit_only_score_days']) or 'none'}.",
            "",
            "## Selector metrics",
            "",
            "```csv",
            summary.to_csv(index=False, float_format="%.8f").rstrip(),
            "```",
            "",
            "## Paired delta: weighted minus equal", "",
            "```json",
            json.dumps({"all_days": paired_all, "changed_selection_days": paired_changed}, ensure_ascii=False, indent=2, default=_json_default),
            "```",
            "",
            "## Weight audit", "",
            "```json",
            json.dumps(weight_meta, ensure_ascii=False, indent=2, default=_json_default),
            "```",
            "",
            "## Current live decision reproduction", "",
            "```json",
            json.dumps(reference, ensure_ascii=False, indent=2, default=_json_default),
            "```",
            "",
            "## Caveats", "",
            "- This is a selector-level replay over frozen active shadow histories, not a new model-training backtest.",
            "- The existing state history carries its prior 14:30 provenance; this test does not establish strict 14:29 PIT certification.",
            "- One fixed kernel is tested. Do not promote it or tune its bandwidth from this result without a separate walk-forward protocol.",
        ]
    )
    return "\n".join(rows) + "\n"


def _write_task_state(run_root: Path, *, status: str, details: dict[str, Any]) -> None:
    lines = [
        "# Task State",
        "",
        "## Objective",
        "",
        "- Research-only comparison of equal-weight versus distance-weighted BaseGap.",
        "",
        "## Risk Level",
        "",
        "- medium: selector semantics are examined, but all inputs/outputs are isolated from live runtime.",
        "",
        "## Current Verified Facts",
        "",
        f"- status: {status}",
        f"- details: `{json.dumps(details, ensure_ascii=False, default=_json_default)}`",
        "",
        "## Files Changed",
        "",
        "- Only files under this research run root.",
        "",
        "## Next Action",
        "",
        "- Review aligned selector metrics before considering any further research; no live promotion is implied.",
    ]
    (run_root / "task_state.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None, help="new leaf directory; existing paths are refused")
    parser.add_argument("--live-config", type=Path, default=DEFAULT_LIVE_CONFIG)
    parser.add_argument("--strategy-config", type=Path, default=DEFAULT_STRATEGY_CONFIG)
    parser.add_argument("--live-decision", type=Path, default=DEFAULT_LIVE_DECISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = _assert_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root / run_name
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing run root: {run_root}")
    run_root.mkdir(parents=False, exist_ok=False)
    _write_task_state(run_root, status="started", details={"run_root": str(run_root)})

    snapshot = _snapshot_inputs(
        run_root=run_root,
        live_config_path=args.live_config,
        strategy_config_path=args.strategy_config,
        live_decision_path=args.live_decision,
    )
    daily, weights, replay_meta = _replay(snapshot)
    equal_metrics = _selector_metrics(
        daily,
        return_column="equal_full_return",
        selected_column="equal_full_selected_model_id",
    )
    weighted_metrics = _selector_metrics(
        daily,
        return_column="weighted_full_return",
        selected_column="weighted_full_selected_model_id",
    )
    summary = pd.DataFrame(
        [
            {"variant": "equal_weight_current_base", **equal_metrics},
            {"variant": "distance_weighted_median_half", **weighted_metrics},
        ]
    )
    paired_all = _paired_metrics(daily, changed_only=False)
    paired_changed = _paired_metrics(daily, changed_only=True)
    weight_meta = _weight_summary(weights)
    reference = _live_reference_check(snapshot, daily)
    nav_path = _write_nav_plot(run_root, daily)

    decision_changes = {
        "base_selected_changed_days": int(daily["base_selected_changed"].sum()),
        "full_selected_changed_days": int(daily["full_selected_changed"].sum()),
        "full_selected_changed_realised_days": int(
            daily.loc[daily["outcome_available"], "full_selected_changed"].sum()
        ),
    }
    final_manifest = {
        **snapshot.input_manifest,
        "fixed_variant": {
            "base_candidate_window": 60,
            "nearest_k": 40,
            "distance_weight": "w_i = 2^(-(d_i / median(d))^2), normalized; equal distances fall back to equal weight",
            "score": "20% fractional-mass weighted trim mean - weighted winsor10/90 std / sqrt(Kish ESS)",
            "unchanged": [
                "state z-score and Euclidean distance",
                "candidate returns and candidate order",
                "BaseGap margin",
                "Champion-first protection",
                "Champion-third veto",
                "Ridge Robust and Fusion routing",
                "strategy/mask/cost/benchmark shadow-return contract",
            ],
        },
        "replay": replay_meta,
        "decision_changes": decision_changes,
        "metrics": {
            "equal_weight_current_base": equal_metrics,
            "distance_weighted_median_half": weighted_metrics,
            "weighted_minus_equal_all_days": paired_all,
            "weighted_minus_equal_changed_selection_days": paired_changed,
        },
        "weight_audit": weight_meta,
        "live_reference_reproduction": reference,
        "artifacts": {
            "daily_selector_replay": "daily_selector_replay.csv",
            "daily_weight_audit": "daily_weight_audit.csv",
            "summary_metrics": "summary_metrics.csv",
            "summary": "summary.md",
            "nav_plot": nav_path.name,
        },
    }
    daily.to_csv(run_root / "daily_selector_replay.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(run_root / "daily_weight_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(run_root / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    (run_root / "input_manifest.json").write_text(
        json.dumps(snapshot.input_manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (run_root / "summary.json").write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (run_root / "summary.md").write_text(
        _summary_markdown(
            summary=summary,
            paired_all=paired_all,
            paired_changed=paired_changed,
            replay_meta=replay_meta,
            weight_meta=weight_meta,
            reference=reference,
        ),
        encoding="utf-8",
    )
    _write_task_state(run_root, status="complete", details={"decision_changes": decision_changes, "metrics": final_manifest["metrics"]})
    print(f"OUTPUT_ROOT {run_root}")
    print(f"SUMMARY {run_root / 'summary.md'}")
    print(f"FULL_SELECTION_CHANGES {decision_changes['full_selected_changed_realised_days']}")


if __name__ == "__main__":
    main()
