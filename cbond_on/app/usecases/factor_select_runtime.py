from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.app.usecases.model_score_runtime import run as run_model_score
from cbond_on.infra.factors.quality import expected_factor_columns_from_cfg
from cbond_on.infra.model.eval.evaluator import (
    EvaluationResult,
    evaluate_merged_scores,
    load_scores_frame,
    merge_score_with_label,
)


@dataclass
class _TrialResult:
    factor: str
    status: str
    trial_dir: Path
    score_output: Path
    summary: dict[str, Any]
    daily: pd.DataFrame
    uplift: dict[str, Any]
    error: str = ""


def _safe_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "empty"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _load_factor_list(path_like: str) -> list[str]:
    path = resolve_config_path(path_like)
    payload = load_json_like(path)
    raw: Any
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("factors", [])
    else:
        raw = []
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            name = str(item).strip()
            if name:
                out.append(name)
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _t_stat_from_series(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(s))
    if n < 2:
        return float("nan")
    std = float(s.std(ddof=1))
    if std <= 0:
        return float("nan")
    mean = float(s.mean())
    return float(mean / (std / math.sqrt(n)))


def _calc_uplift(baseline_daily: pd.DataFrame, candidate_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = baseline_daily[["trade_date", "rank_ic", "ic", "dir"]].copy()
    cand = candidate_daily[["trade_date", "rank_ic", "ic", "dir"]].copy()
    base = base.rename(
        columns={
            "rank_ic": "rank_ic_base",
            "ic": "ic_base",
            "dir": "dir_base",
        }
    )
    cand = cand.rename(
        columns={
            "rank_ic": "rank_ic_cand",
            "ic": "ic_cand",
            "dir": "dir_cand",
        }
    )
    merged = base.merge(cand, on="trade_date", how="inner")
    if merged.empty:
        return merged, {
            "days_overlap": 0,
            "delta_rank_ic_mean": float("nan"),
            "delta_rank_ic_t": float("nan"),
            "delta_rank_ic_win_rate": float("nan"),
            "delta_ic_mean": float("nan"),
            "delta_ic_t": float("nan"),
            "delta_dir_mean": float("nan"),
        }

    merged["delta_rank_ic"] = pd.to_numeric(merged["rank_ic_cand"], errors="coerce") - pd.to_numeric(
        merged["rank_ic_base"], errors="coerce"
    )
    merged["delta_ic"] = pd.to_numeric(merged["ic_cand"], errors="coerce") - pd.to_numeric(
        merged["ic_base"], errors="coerce"
    )
    merged["delta_dir"] = pd.to_numeric(merged["dir_cand"], errors="coerce") - pd.to_numeric(
        merged["dir_base"], errors="coerce"
    )

    d_rank = pd.to_numeric(merged["delta_rank_ic"], errors="coerce").dropna()
    d_ic = pd.to_numeric(merged["delta_ic"], errors="coerce").dropna()
    d_dir = pd.to_numeric(merged["delta_dir"], errors="coerce").dropna()

    uplift = {
        "days_overlap": int(len(merged)),
        "delta_rank_ic_mean": float(d_rank.mean()) if not d_rank.empty else float("nan"),
        "delta_rank_ic_t": _t_stat_from_series(d_rank),
        "delta_rank_ic_win_rate": float((d_rank > 0).mean()) if not d_rank.empty else float("nan"),
        "delta_ic_mean": float(d_ic.mean()) if not d_ic.empty else float("nan"),
        "delta_ic_t": _t_stat_from_series(d_ic),
        "delta_dir_mean": float(d_dir.mean()) if not d_dir.empty else float("nan"),
    }
    return merged, uplift


def _plot_candidate_report(
    *,
    factor: str,
    out_path: Path,
    baseline_daily: pd.DataFrame,
    candidate_daily: pd.DataFrame,
    uplift_daily: pd.DataFrame,
    uplift: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    b = baseline_daily.copy()
    c = candidate_daily.copy()
    if not b.empty:
        b["trade_date"] = pd.to_datetime(b["trade_date"], errors="coerce")
        ax.plot(b["trade_date"], pd.to_numeric(b["rank_ic"], errors="coerce"), label="baseline_rank_ic")
    if not c.empty:
        c["trade_date"] = pd.to_datetime(c["trade_date"], errors="coerce")
        ax.plot(c["trade_date"], pd.to_numeric(c["rank_ic"], errors="coerce"), label="candidate_rank_ic")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Daily RankIC")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    if not uplift_daily.empty:
        u = uplift_daily.copy()
        u["trade_date"] = pd.to_datetime(u["trade_date"], errors="coerce")
        d = pd.to_numeric(u["delta_rank_ic"], errors="coerce")
        cum = d.fillna(0.0).cumsum()
        ax.plot(u["trade_date"], cum, color="#4C78A8", label="cum(delta_rank_ic)")
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Cumulative Delta RankIC")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if not uplift_daily.empty:
        vals = pd.to_numeric(uplift_daily["delta_rank_ic"], errors="coerce").dropna()
        if not vals.empty:
            ax.hist(vals, bins=40, color="#72B7B2", alpha=0.85)
            ax.axvline(float(vals.mean()), color="#E45756", linestyle="--", linewidth=1.2, label="mean")
            ax.legend(loc="best", fontsize=8)
    ax.set_title("Delta RankIC Distribution")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    text_lines = [
        f"factor: {factor}",
        f"days_overlap: {uplift.get('days_overlap')}",
        f"delta_rank_ic_mean: {uplift.get('delta_rank_ic_mean'):.6f}" if pd.notna(uplift.get("delta_rank_ic_mean")) else "delta_rank_ic_mean: nan",
        f"delta_rank_ic_t: {uplift.get('delta_rank_ic_t'):.4f}" if pd.notna(uplift.get("delta_rank_ic_t")) else "delta_rank_ic_t: nan",
        f"delta_rank_ic_win_rate: {uplift.get('delta_rank_ic_win_rate'):.4f}" if pd.notna(uplift.get("delta_rank_ic_win_rate")) else "delta_rank_ic_win_rate: nan",
        f"delta_ic_mean: {uplift.get('delta_ic_mean'):.6f}" if pd.notna(uplift.get("delta_ic_mean")) else "delta_ic_mean: nan",
        f"delta_ic_t: {uplift.get('delta_ic_t'):.4f}" if pd.notna(uplift.get("delta_ic_t")) else "delta_ic_t: nan",
        f"delta_dir_mean: {uplift.get('delta_dir_mean'):.6f}" if pd.notna(uplift.get("delta_dir_mean")) else "delta_dir_mean: nan",
    ]
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_trial_artifacts(
    *,
    trial: _TrialResult,
    uplift_daily: pd.DataFrame,
    baseline_summary: dict[str, Any],
    baseline_daily: pd.DataFrame,
) -> None:
    trial.trial_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "factor": trial.factor,
        "status": trial.status,
        "error": trial.error,
        "summary": trial.summary,
        "uplift": trial.uplift,
        "baseline_summary": baseline_summary,
    }
    (trial.trial_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not trial.daily.empty:
        trial.daily.to_csv(trial.trial_dir / "evaluation_daily.csv", index=False)
    if not uplift_daily.empty:
        uplift_daily.to_csv(trial.trial_dir / "uplift_daily.csv", index=False)
    _plot_candidate_report(
        factor=trial.factor,
        out_path=trial.trial_dir / "factor_uplift_report.png",
        baseline_daily=baseline_daily,
        candidate_daily=trial.daily,
        uplift_daily=uplift_daily,
        uplift=trial.uplift,
    )


def _evaluate_score_output(
    *,
    score_output: Path,
    label_root: Path,
    factor_time: str,
    label_time: str,
    start_day: date,
    end_day: date,
    bins: int,
) -> EvaluationResult:
    scores = load_scores_frame(score_output)
    merged = merge_score_with_label(
        scores=scores,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        start=start_day,
        end=end_day,
    )
    return evaluate_merged_scores(merged, bins=bins)


def _build_trial_model_cfg(
    *,
    base_model_cfg: dict[str, Any],
    start_day: date,
    end_day: date,
    factors: list[str],
    model_name: str,
    score_output: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_model_cfg)
    cfg["start"] = str(start_day)
    cfg["end"] = str(end_day)
    cfg["model_name"] = model_name
    cfg["factors"] = list(factors)
    cfg["score_output"] = str(score_output.as_posix())
    cfg["score_overwrite"] = True
    cfg["score_dedupe"] = True
    inc = dict(cfg.get("incremental", {}))
    inc["enabled"] = True
    inc["skip_existing_scores"] = False
    inc["warm_start"] = False
    inc["save_state"] = False
    cfg["incremental"] = inc
    return cfg


def run(
    *,
    cfg: dict | None = None,
    config_name: str | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
) -> dict[str, Any]:
    selector_cfg = dict(cfg or load_config_file(config_name or "score/factor_select"))
    paths_cfg = load_config_file("paths")
    score_cfg = load_config_file(str(selector_cfg.get("model_score_config", "score/model_score")))
    factor_cfg = load_config_file(str(selector_cfg.get("factor_config", "factor")))

    model_id = str(
        selector_cfg.get("base_model_id")
        or score_cfg.get("model_id")
        or score_cfg.get("default_model_id", "")
    ).strip()
    if not model_id:
        raise ValueError("factor_select missing base_model_id")
    models = dict(score_cfg.get("models", {}))
    if model_id not in models:
        raise KeyError(f"base_model_id not found in model_score config: {model_id}")
    model_entry = dict(models[model_id])
    model_type = str(model_entry.get("model_type", "")).strip()
    model_cfg_key = str(model_entry.get("model_config", "")).strip()
    if not model_type or not model_cfg_key:
        raise ValueError(f"invalid model entry for base model: {model_id}")

    base_model_cfg_path = resolve_config_path(model_cfg_key)
    base_model_cfg = load_json_like(base_model_cfg_path)

    start_day = parse_date(start or selector_cfg.get("start") or base_model_cfg.get("start"))
    end_day = parse_date(end or selector_cfg.get("end") or base_model_cfg.get("end"))
    if start_day > end_day:
        raise ValueError("start must be <= end")

    label_root = Path(paths_cfg["label_data_root"])
    factor_time = str(base_model_cfg.get("factor_time", "14:30"))
    label_time = str(base_model_cfg.get("label_time", "14:42"))
    bins = int(selector_cfg.get("bins", base_model_cfg.get("bins", 5)))
    execution_override = dict(selector_cfg.get("execution", {}))

    baseline_factors = _load_factor_list(str(selector_cfg.get("baseline_factors_file", "score/factor_baseline_factors")))
    if not baseline_factors:
        raise ValueError("baseline factor list is empty")
    blacklist = set(_load_factor_list(str(selector_cfg.get("blacklist_file", "score/factor_blacklist"))))

    candidate_all = expected_factor_columns_from_cfg(factor_cfg)
    baseline_set = set(baseline_factors)
    candidates = [x for x in candidate_all if x not in baseline_set and x not in blacklist]
    max_candidates = int(selector_cfg.get("max_candidates", 0) or 0)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    results_root = Path(paths_cfg["results_root"])
    date_label = f"{start_day:%Y-%m-%d}_{end_day:%Y-%m-%d}"
    out_root = results_root / date_label / "Factor_Select" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    candidates_root = out_root / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)

    acceptance_cfg = dict(selector_cfg.get("acceptance", {}))
    min_days = int(acceptance_cfg.get("min_days", 20))
    min_delta_rank_ic_mean = float(acceptance_cfg.get("min_delta_rank_ic_mean", 0.0))
    min_t_stat = float(acceptance_cfg.get("min_t_stat", 2.0))
    min_win_rate = float(acceptance_cfg.get("min_win_rate", 0.55))

    # baseline
    baseline_dir = out_root / "baseline"
    baseline_score_output = baseline_dir / "scores"
    baseline_cfg = _build_trial_model_cfg(
        base_model_cfg=base_model_cfg,
        start_day=start_day,
        end_day=end_day,
        factors=baseline_factors,
        model_name=f"{base_model_cfg.get('model_name', model_id)}__factor_select_baseline",
        score_output=baseline_score_output,
    )
    baseline_cfg_path = baseline_dir / "model_config.json"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_cfg_path.write_text(json.dumps(baseline_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline_score_cfg = copy.deepcopy(score_cfg)
    baseline_score_cfg["model_id"] = "factor_select_baseline"
    baseline_score_cfg["default_model_id"] = "factor_select_baseline"
    baseline_score_cfg["models"] = {
        "factor_select_baseline": {
            "model_type": model_type,
            "model_config": str(baseline_cfg_path.as_posix()),
        }
    }
    baseline_score_cfg["execution"] = execution_override
    print(f"[factor_select] baseline start model={model_id} factors={len(baseline_factors)}")
    run_model_score(
        model_id="factor_select_baseline",
        start=start_day,
        end=end_day,
        cfg=baseline_score_cfg,
    )
    baseline_eval = _evaluate_score_output(
        score_output=baseline_score_output,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        start_day=start_day,
        end_day=end_day,
        bins=bins,
    )
    baseline_summary = dict(baseline_eval.summary)
    (baseline_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "factor_count": len(baseline_factors),
                "summary": baseline_eval.summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not baseline_eval.daily.empty:
        baseline_eval.daily.to_csv(baseline_dir / "evaluation_daily.csv", index=False)

    # candidate loop
    rows: list[dict[str, Any]] = []
    accepted: list[str] = []
    total = len(candidates)
    for idx, factor in enumerate(candidates, start=1):
        print(f"[factor_select] {idx}/{total} factor={factor} start")
        trial_dir = candidates_root / _safe_name(factor)
        score_output = trial_dir / "scores"
        trial = _TrialResult(
            factor=factor,
            status="ok",
            trial_dir=trial_dir,
            score_output=score_output,
            summary={},
            daily=pd.DataFrame(),
            uplift={},
            error="",
        )
        uplift_daily = pd.DataFrame()
        try:
            trial_cfg = _build_trial_model_cfg(
                base_model_cfg=base_model_cfg,
                start_day=start_day,
                end_day=end_day,
                factors=baseline_factors + [factor],
                model_name=f"{base_model_cfg.get('model_name', model_id)}__factor_select__{_safe_name(factor)}",
                score_output=score_output,
            )
            trial_cfg_path = trial_dir / "model_config.json"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_cfg_path.write_text(json.dumps(trial_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

            trial_score_cfg = copy.deepcopy(score_cfg)
            trial_score_cfg["model_id"] = "factor_select_trial"
            trial_score_cfg["default_model_id"] = "factor_select_trial"
            trial_score_cfg["models"] = {
                "factor_select_trial": {
                    "model_type": model_type,
                    "model_config": str(trial_cfg_path.as_posix()),
                }
            }
            trial_score_cfg["execution"] = execution_override

            run_model_score(
                model_id="factor_select_trial",
                start=start_day,
                end=end_day,
                cfg=trial_score_cfg,
            )
            eval_res = _evaluate_score_output(
                score_output=score_output,
                label_root=label_root,
                factor_time=factor_time,
                label_time=label_time,
                start_day=start_day,
                end_day=end_day,
                bins=bins,
            )
            trial.summary = dict(eval_res.summary)
            trial.daily = eval_res.daily
            uplift_daily, uplift = _calc_uplift(baseline_eval.daily, eval_res.daily)
            trial.uplift = uplift

            is_accepted = (
                int(uplift.get("days_overlap", 0)) >= min_days
                and pd.notna(uplift.get("delta_rank_ic_mean"))
                and float(uplift.get("delta_rank_ic_mean", float("nan"))) >= min_delta_rank_ic_mean
                and pd.notna(uplift.get("delta_rank_ic_t"))
                and float(uplift.get("delta_rank_ic_t", float("nan"))) >= min_t_stat
                and pd.notna(uplift.get("delta_rank_ic_win_rate"))
                and float(uplift.get("delta_rank_ic_win_rate", float("nan"))) >= min_win_rate
            )
            if is_accepted:
                accepted.append(factor)
            rows.append(
                {
                    "factor": factor,
                    "status": "ok",
                    "accepted": bool(is_accepted),
                    **{f"cand_{k}": v for k, v in trial.summary.items()},
                    **trial.uplift,
                    "error": "",
                }
            )
            print(
                f"[factor_select] {idx}/{total} factor={factor} done "
                f"delta_rank_ic_mean={trial.uplift.get('delta_rank_ic_mean')} "
                f"t={trial.uplift.get('delta_rank_ic_t')}"
            )
        except Exception as exc:
            trial.status = "failed"
            trial.error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "factor": factor,
                    "status": "failed",
                    "accepted": False,
                    "error": trial.error,
                }
            )
            print(f"[factor_select] {idx}/{total} factor={factor} failed: {trial.error}")
        finally:
            _save_trial_artifacts(
                trial=trial,
                uplift_daily=uplift_daily,
                baseline_summary=baseline_summary,
                baseline_daily=baseline_eval.daily,
            )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        sort_cols = [c for c in ["accepted", "delta_rank_ic_t", "delta_rank_ic_mean"] if c in summary_df.columns]
        if sort_cols:
            summary_df = summary_df.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")
        summary_df.to_csv(out_root / "factor_uplift_summary.csv", index=False)

    accepted_payload = {
        "model_id": model_id,
        "start": str(start_day),
        "end": str(end_day),
        "baseline_factor_count": len(baseline_factors),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "accepted_factors": accepted,
        "acceptance": {
            "min_days": min_days,
            "min_delta_rank_ic_mean": min_delta_rank_ic_mean,
            "min_t_stat": min_t_stat,
            "min_win_rate": min_win_rate,
        },
    }
    (out_root / "accepted_factors.json").write_text(
        json.dumps(accepted_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_root / "run_config_snapshot.json").write_text(
        json.dumps(
            {
                "selector_config": selector_cfg,
                "model_score_config_key": str(selector_cfg.get("model_score_config", "score/model_score")),
                "base_model_config_path": str(base_model_cfg_path.as_posix()),
                "baseline_factors_file": str(selector_cfg.get("baseline_factors_file")),
                "blacklist_file": str(selector_cfg.get("blacklist_file")),
                "baseline_factors": baseline_factors,
                "blacklist": sorted(list(blacklist)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "out_root": str(out_root.as_posix()),
        "model_id": model_id,
        "baseline_factors": len(baseline_factors),
        "candidate_factors": len(candidates),
        "accepted_factors": len(accepted),
    }
