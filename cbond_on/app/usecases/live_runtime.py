from __future__ import annotations

from dataclasses import replace
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.trading_days import prev_trading_days_from_raw
from cbond_on.infra.data.io import read_clean_daily
from cbond_on.infra.universe.pool_filter import (
    apply_allowlist_filter_to_universe,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)
from cbond_on.domain.signals.service import SignalSelectionRequest, select_signals
from cbond_on.infra.model.score_io import load_scores_by_date, write_scores_by_date
from cbond_on.app.usecases.factor_build_runtime import run as run_factor_build
from cbond_on.app.usecases.label_runtime import run as run_label_build
from cbond_on.app.usecases.model_score_runtime import run as run_model_score
from cbond_on.app.usecases.panel_runtime import run as run_panel_build
from cbond_on.infra.live.config import (
    assert_no_date_fields_in_live_config,
    load_live_factor_runtime,
    load_live_model_runtime,
    load_strategy_config,
)
from cbond_on.infra.live.db_writer import write_trades_to_db
from cbond_on.infra.live.holdings import load_previous_holdings
from cbond_on.infra.live.model_switch import (
    SwitchDecision,
    build_rank_average_scores,
    decide_scoreopt_bm_short,
    decide_scoreopt_t1430_dispersion,
    decide_scoreopt_t1430_fusion_gate,
    decide_single_challenger_by_regime,
    decide_single_challenger_by_sharpe,
    update_t1430_market_state_feature_history,
    write_switch_decision,
)
from cbond_on.infra.live.shadow_returns import (
    ShadowReturnUpdateResult,
    update_shadow_return_history,
)
from cbond_on.infra.live.publish_gate import (
    data_hub_runtime_from_live,
    ensure_publish_ready,
    today_shanghai,
)
from cbond_on.infra.live.score import resolve_score_df_for_target


O005_ALLOWLIST_TABLE = "quant_factor_dev.researcher_xuvb.o_0005"

_LIVE_MODEL_SWITCH_FALLBACK_REASONS = frozenset(
    {
        "feature_missing",
        "feature_na",
        "insufficient_history",
        "insufficient_history_or_zero_volatility",
        "margin_default",
    }
)
_LIVE_MODEL_SWITCH_FALLBACK_PREFIXES = ("fallback_rolling_sharpe_",)


def _month_day_path(root: str | Path, day: date, *, filename_root: str = "") -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    base = Path(root)
    return base / filename_root / month / filename if filename_root else base / month / filename


def _panel_day_path(paths_cfg: dict, day: date, *, asset: str, panel_name: str) -> Path:
    return (
        Path(paths_cfg["panel_data_root"])
        / "panels"
        / asset
        / panel_name
        / f"{day.year:04d}-{day.month:02d}"
        / f"{day.strftime('%Y%m%d')}.parquet"
    )


def _factor_day_path(paths_cfg: dict, day: date, *, panel_name: str) -> Path:
    return (
        Path(paths_cfg["factor_data_root"])
        / "factors"
        / panel_name
        / f"{day.year:04d}-{day.month:02d}"
        / f"{day.strftime('%Y%m%d')}.parquet"
    )


def _label_day_path(paths_cfg: dict, day: date) -> Path:
    return _month_day_path(paths_cfg["label_data_root"], day)


def _require_existing(path: Path, *, name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{name} missing after live build: {path}")


def _normalize_assets(value: object) -> list[str]:
    if isinstance(value, str):
        return [x.strip().lower() for x in value.replace(";", ",").split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip().lower() for x in value if str(x).strip()]
    return []


def _factor_panel_source_mode(factor_cfg: dict) -> str:
    raw = factor_cfg.get("panel_source")
    if isinstance(raw, dict):
        raw = raw.get("mode", "cached_panel")
    text = str(raw or "cached_panel").strip().lower()
    if text in {"clean", "clean_data", "clean_direct", "on_demand", "on_demand_clean"}:
        return "clean_direct"
    return "cached_panel"


def _missing_days(days: list[date], *, path_builder) -> list[date]:
    return [day for day in days if not path_builder(day).exists()]


def _build_day_span(days: list[date]) -> tuple[date, date]:
    ordered = sorted(set(days))
    return ordered[0], ordered[-1]


def _parse_live_model_window_days(live_model_score_cfg: dict, model_id: str) -> int:
    models_raw = live_model_score_cfg.get("models", {})
    if not isinstance(models_raw, dict):
        return 0
    model_entry_raw = models_raw.get(model_id, {})
    if not isinstance(model_entry_raw, dict):
        return 0
    model_cfg_key = str(model_entry_raw.get("model_config", "")).strip()
    if not model_cfg_key:
        return 0
    model_cfg = load_json_like(resolve_config_path(model_cfg_key))
    rolling_cfg = model_cfg.get("rolling", {})
    if not isinstance(rolling_cfg, dict):
        return 0
    if not bool(rolling_cfg.get("enabled", False)):
        return 0
    return int(rolling_cfg.get("window_days", 0))


def _resolve_model_result_score_path(
    model_result: dict,
    *,
    model_id: str,
    paths_cfg: dict,
) -> Path:
    return resolve_output_path(
        model_result.get("score_output"),
        default_path=Path(paths_cfg["results_root"]) / "scores" / model_id,
        results_root=paths_cfg["results_root"],
    )


def _resolve_switch_score_output(
    raw_path: object,
    *,
    model_id: str,
    paths_cfg: dict,
) -> Path:
    return resolve_output_path(
        raw_path,
        default_path=Path(paths_cfg["results_root"]) / "scores" / "live" / model_id,
        results_root=paths_cfg["results_root"],
    )


def _resolve_switch_return_path(raw_path: object, *, paths_cfg: dict) -> str:
    return str(
        resolve_output_path(
            raw_path,
            default_path=Path(paths_cfg["results_root"]) / "analysis" / "model_switch_return_history_missing.csv",
            results_root=paths_cfg["results_root"],
        )
    )


def _resolve_model_switch_return_paths(switch_cfg: dict, *, paths_cfg: dict) -> dict:
    cfg = dict(switch_cfg)

    def _resolve_group(group_raw: object) -> dict | object:
        if not isinstance(group_raw, dict):
            return group_raw
        group = dict(group_raw)
        for path_key in ("return_path", "score_return_path"):
            if path_key in group:
                group[path_key] = _resolve_switch_return_path(group.get(path_key), paths_cfg=paths_cfg)
        return group

    for key in ("champion", "challenger"):
        cfg[key] = _resolve_group(cfg.get(key, {}))
    challengers_raw = cfg.get("challengers")
    if isinstance(challengers_raw, list):
        cfg["challengers"] = [_resolve_group(item) for item in challengers_raw]
    if "state_feature_path" in cfg:
        cfg["state_feature_path"] = str(
            resolve_output_path(
                cfg.get("state_feature_path"),
                default_path=Path(paths_cfg["results_root"]) / "analysis" / "model_switch_t1430_state_features.csv",
                results_root=paths_cfg["results_root"],
            )
        )
    return cfg


def _score_df_from_path(score_path: Path, score_day: date) -> pd.DataFrame:
    score_cache = load_scores_by_date(score_path)
    return resolve_score_df_for_target(score_cache, score_day, score_path)


def _run_switch_source_score(
    *,
    source_cfg: dict,
    score_day: date,
    label_cutoff: date,
    paths_cfg: dict,
) -> tuple[str, str, Path, pd.DataFrame]:
    model_id = str(source_cfg.get("model_id", "")).strip()
    if not model_id:
        raise ValueError("model_switch challenger source missing model_id")
    config_key = str(source_cfg.get("config", "")).strip()
    if not config_key:
        raise ValueError(f"model_switch challenger source missing config: {model_id}")
    source_name = str(source_cfg.get("name") or model_id).strip()
    score_cfg = dict(load_config_file(config_key))
    print(
        "live model switch source score:",
        f"name={source_name}",
        f"model_id={model_id}",
        f"config={config_key}",
    )
    result = run_model_score(
        model_id=model_id,
        start=score_day,
        end=score_day,
        label_cutoff=label_cutoff,
        cfg=score_cfg,
    )
    score_path = _resolve_model_result_score_path(result, model_id=model_id, paths_cfg=paths_cfg)
    score_df = _score_df_from_path(score_path, score_day)
    return model_id, source_name, score_path, score_df


def _build_switch_challenger_score(
    *,
    switch_cfg: dict,
    challenger_cfg: dict | None = None,
    current_model_id: str,
    current_score_path: Path,
    current_score_df: pd.DataFrame,
    score_day: date,
    label_cutoff: date,
    paths_cfg: dict,
) -> tuple[str, Path, pd.DataFrame, list[dict]]:
    challenger_cfg = dict(challenger_cfg or switch_cfg.get("challenger", {}))
    challenger_model_id = str(challenger_cfg.get("model_id", "")).strip()
    if not challenger_model_id:
        raise ValueError("model_switch.challenger.model_id must not be empty")
    kind = str(challenger_cfg.get("kind", "rankavg")).strip().lower()
    if kind in {"single", "model", "direct"}:
        if challenger_model_id == current_model_id:
            source_name = str(challenger_cfg.get("name") or challenger_model_id).strip()
            return (
                challenger_model_id,
                current_score_path,
                current_score_df[["code", "score"]].copy(),
                [
                    {
                        "name": source_name,
                        "model_id": challenger_model_id,
                        "score_path": str(current_score_path),
                        "rows": int(len(current_score_df)),
                    }
                ],
            )
        source_model_id, source_name, score_path, score_df = _run_switch_source_score(
            source_cfg=challenger_cfg,
            score_day=score_day,
            label_cutoff=label_cutoff,
            paths_cfg=paths_cfg,
        )
        if source_model_id != challenger_model_id:
            raise ValueError(
                "model_switch single challenger source model mismatch: "
                f"challenger={challenger_model_id}, source={source_model_id}"
            )
        return (
            challenger_model_id,
            score_path,
            score_df[["code", "score"]].copy(),
            [
                {
                    "name": source_name,
                    "model_id": source_model_id,
                    "score_path": str(score_path),
                    "rows": int(len(score_df)),
                }
            ],
        )

    if kind not in {"rankavg", "rank_average"}:
        raise ValueError(f"unsupported live challenger kind: {kind}")
    sources_raw = challenger_cfg.get("sources", [])
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError("model_switch.challenger.sources must be a non-empty list")

    available: dict[str, tuple[str, Path, pd.DataFrame]] = {
        current_model_id: (
            str(switch_cfg.get("champion", {}).get("name") or current_model_id),
            current_score_path,
            current_score_df,
        )
    }
    rank_frames: list[tuple[str, pd.DataFrame]] = []
    source_details: list[dict] = []

    for source_raw in sources_raw:
        if not isinstance(source_raw, dict):
            raise ValueError("model_switch.challenger.sources entries must be objects")
        source = dict(source_raw)
        source_model_id = str(source.get("model_id", "")).strip()
        if not source_model_id:
            raise ValueError("model_switch challenger source missing model_id")
        if source_model_id in available:
            source_name, score_path, score_df = available[source_model_id]
        else:
            source_model_id, source_name, score_path, score_df = _run_switch_source_score(
                source_cfg=source,
                score_day=score_day,
                label_cutoff=label_cutoff,
                paths_cfg=paths_cfg,
            )
            available[source_model_id] = (source_name, score_path, score_df)

        rank_frames.append((source_name, score_df))
        source_details.append(
            {
                "name": source_name,
                "model_id": source_model_id,
                "score_path": str(score_path),
                "rows": int(len(score_df)),
            }
        )

    challenger_score_df = build_rank_average_scores(rank_frames, score_day=score_day)
    challenger_output = _resolve_switch_score_output(
        challenger_cfg.get("score_output"),
        model_id=challenger_model_id,
        paths_cfg=paths_cfg,
    )
    write_scores_by_date(
        challenger_output,
        challenger_score_df,
        overwrite=False,
        dedupe=True,
    )
    return challenger_model_id, challenger_output, challenger_score_df[["code", "score"]].copy(), source_details


def _switch_challenger_configs(switch_cfg: dict) -> list[dict]:
    challengers_raw = switch_cfg.get("challengers")
    if isinstance(challengers_raw, list) and challengers_raw:
        return [dict(item) for item in challengers_raw if isinstance(item, dict)]
    challenger = dict(switch_cfg.get("challenger", {}))
    return [challenger] if challenger else []


def _build_switch_challenger_scores(
    *,
    switch_cfg: dict,
    current_model_id: str,
    current_score_path: Path,
    current_score_df: pd.DataFrame,
    score_day: date,
    label_cutoff: date,
    paths_cfg: dict,
) -> list[dict]:
    results: list[dict] = []
    seen_model_ids: set[str] = set()
    for challenger_cfg in _switch_challenger_configs(switch_cfg):
        challenger_model_id, challenger_score_path, challenger_score_df, source_details = _build_switch_challenger_score(
            switch_cfg=switch_cfg,
            challenger_cfg=challenger_cfg,
            current_model_id=current_model_id,
            current_score_path=current_score_path,
            current_score_df=current_score_df,
            score_day=score_day,
            label_cutoff=label_cutoff,
            paths_cfg=paths_cfg,
        )
        if challenger_model_id in seen_model_ids:
            raise ValueError(f"duplicate model_switch challenger model_id: {challenger_model_id}")
        seen_model_ids.add(challenger_model_id)
        results.append(
            {
                "model_id": challenger_model_id,
                "name": str(challenger_cfg.get("name") or challenger_model_id),
                "score_path": challenger_score_path,
                "score_df": challenger_score_df,
                "return_path": challenger_cfg.get("return_path"),
                "source_scores": source_details,
            }
        )
    if not results:
        raise ValueError("model_switch requires at least one challenger")
    return results


def _live_model_switch_fallback_reason(decision: SwitchDecision) -> str | None:
    reason = str(decision.reason or "").strip()
    if reason in _LIVE_MODEL_SWITCH_FALLBACK_REASONS:
        return reason
    if any(reason.startswith(prefix) for prefix in _LIVE_MODEL_SWITCH_FALLBACK_PREFIXES):
        return reason
    if reason == "fusion_base_robust_not_confident":
        base_reason = ""
        if isinstance(decision.fusion, dict):
            base = decision.fusion.get("base")
            if isinstance(base, dict):
                base_reason = str(base.get("reason") or "").strip()
        fallback_reason = base_reason or str(decision.fallback_reason or "").strip()
        return f"{reason}:{fallback_reason}" if fallback_reason else reason
    return None


def _assert_no_live_model_switch_fallback(
    decision: SwitchDecision,
    *,
    switch_cfg: dict,
) -> None:
    _ = switch_cfg
    fallback_reason = _live_model_switch_fallback_reason(decision)
    if fallback_reason is None:
        return
    print(
        "live model switch soft degrade:",
        f"mode={decision.mode}",
        f"score_day={decision.score_day}",
        f"reason={decision.reason}",
        f"fallback_reason={decision.fallback_reason}",
        f"fallback_detail={fallback_reason}",
        f"selected={decision.selected_name}",
        f"history_end={decision.history_end}",
        f"history_days={decision.history_days}",
    )


def _model_switch_warning_message(decision: SwitchDecision, fallback_detail: str) -> str:
    reason = str(decision.reason or "").strip()
    selected = str(decision.selected_name or decision.selected_model_id).strip()
    if reason in {"feature_missing", "feature_na"}:
        return f"模型选择提示：当日T1430状态特征不可用，已按规则选择 {selected}，实盘继续写库。"
    if reason in {"insufficient_history", "insufficient_history_or_zero_volatility"}:
        return f"模型选择提示：历史样本不足，已按规则选择 {selected}，实盘继续写库。"
    if reason == "fusion_base_robust_not_confident":
        return f"模型选择提示：base与robust均未形成高置信切换信号，已按规则选择 {selected}，实盘继续写库。"
    if reason == "margin_default":
        return f"模型选择提示：模型分数差距低于切换门槛，已按规则选择 {selected}，实盘继续写库。"
    if reason.startswith("fallback_rolling_sharpe_"):
        return f"模型选择提示：状态样本不足，已使用滚动Sharpe规则选择 {selected}，实盘继续写库。"
    return f"模型选择提示：触发软降级 {fallback_detail}，已选择 {selected}，实盘继续写库。"


def _model_switch_warnings(decision: SwitchDecision) -> list[dict]:
    fallback_detail = _live_model_switch_fallback_reason(decision)
    if fallback_detail is None:
        return []
    return [
        {
            "level": "warning",
            "category": "model_switch_soft_degrade",
            "reason": str(decision.reason or ""),
            "fallback_reason": None if decision.fallback_reason is None else str(decision.fallback_reason),
            "fallback_detail": fallback_detail,
            "selected_model_id": str(decision.selected_model_id),
            "selected_name": str(decision.selected_name),
            "message": _model_switch_warning_message(decision, fallback_detail),
        }
    ]


def _append_live_day_notes(
    *,
    paths_cfg: dict,
    day: date,
    warnings: list[dict],
) -> None:
    if not warnings:
        return
    path = Path(paths_cfg["results_root"]) / "live" / "scheduler" / "dashboard_notes.json"
    day_key = f"{day:%Y-%m-%d}"
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            store = raw if isinstance(raw, dict) else {"days": {}}
        else:
            store = {"days": {}}
        days = store.setdefault("days", {})
        items = days.setdefault(day_key, [])
        if not isinstance(items, list):
            items = []
            days[day_key] = items
        existing_ids = {str(item.get("id", "")) for item in items if isinstance(item, dict)}
        now = datetime.now().isoformat(timespec="seconds")
        appended = False
        for warning in warnings:
            note_id = (
                f"live_model_switch:{day_key}:"
                f"{warning.get('selected_model_id')}:{warning.get('fallback_detail')}"
            )
            if note_id in existing_ids:
                continue
            items.append(
                {
                    "id": note_id,
                    "time": now,
                    "author": "live",
                    "text": str(warning.get("message") or ""),
                }
            )
            appended = True
        if appended:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
    except Exception as exc:
        print(f"live model switch warning note skipped: {type(exc).__name__}: {exc}")


def _apply_live_model_switch(
    *,
    switch_cfg: dict,
    current_model_id: str,
    current_score_path: Path,
    current_score_df: pd.DataFrame,
    score_day: date,
    expected_history_end: date | None,
    label_cutoff: date,
    raw_root: str,
    clean_root: str,
    strategy_id: str,
    strategy_config: dict,
    allowlist_cfg: dict,
    paths_cfg: dict,
) -> tuple[str, pd.DataFrame, SwitchDecision, dict]:
    switch_cfg = _resolve_model_switch_return_paths(switch_cfg, paths_cfg=paths_cfg)
    mode = str(switch_cfg.get("mode", "single_challenger")).strip().lower()
    if mode not in {
        "single_challenger",
        "regime_bm20_sign",
        "scoreopt_bm_short",
        "scoreopt_t1430_dispersion",
        "scoreopt_t1430_fusion_gate",
    }:
        raise ValueError(f"unsupported live model_switch.mode: {mode}")
    champion_cfg = dict(switch_cfg.get("champion", {}))
    champion_model_id = str(champion_cfg.get("model_id", "")).strip()
    if champion_model_id and champion_model_id != current_model_id:
        raise ValueError(
            "model_switch champion must match live model_score.model_id: "
            f"champion={champion_model_id}, live={current_model_id}"
        )

    challenger_results = _build_switch_challenger_scores(
        switch_cfg=switch_cfg,
        current_model_id=current_model_id,
        current_score_path=current_score_path,
        current_score_df=current_score_df,
        score_day=score_day,
        label_cutoff=label_cutoff,
        paths_cfg=paths_cfg,
    )
    shadow_update_results: list[ShadowReturnUpdateResult] = []
    shadow_cfg = switch_cfg.get("shadow_return_update", {})
    shadow_cfg = shadow_cfg if isinstance(shadow_cfg, dict) else {}
    shadow_enabled = bool(shadow_cfg.get("enabled", True))
    shadow_fail_on_error = bool(shadow_cfg.get("fail_on_error", False))
    if shadow_enabled:
        return_col = str(switch_cfg.get("return_col", "day_return")).strip() or "day_return"
        update_targets = [
            (
                current_model_id,
                current_score_path,
                dict(switch_cfg.get("champion", {})).get("return_path"),
            ),
        ]
        for result in challenger_results:
            update_targets.append(
                (
                    str(result["model_id"]),
                    result["score_path"],
                    result.get("return_path"),
                )
            )
        seen_return_paths: set[str] = set()
        for target_model_id, target_score_path, target_return_path in update_targets:
            if not target_return_path:
                continue
            key = str(target_return_path)
            if key in seen_return_paths:
                continue
            seen_return_paths.add(key)
            try:
                update_result = update_shadow_return_history(
                    model_id=target_model_id,
                    raw_data_root=raw_root,
                    score_path=target_score_path,
                    return_path=target_return_path,
                    expected_history_end=expected_history_end,
                    strategy_id=strategy_id,
                    strategy_config=strategy_config,
                    allowlist_cfg=allowlist_cfg,
                    return_col=return_col,
                )
            except Exception as exc:
                if shadow_fail_on_error:
                    raise
                update_result = ShadowReturnUpdateResult(
                    model_id=target_model_id,
                    score_path=str(target_score_path),
                    return_path=str(target_return_path),
                    expected_history_end=expected_history_end,
                    before_history_end=None,
                    after_history_end=None,
                    appended_rows=0,
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            shadow_update_results.append(update_result)
            print(
                "live model switch shadow return update:",
                f"model_id={update_result.model_id}",
                f"status={update_result.status}",
                f"before={update_result.before_history_end}",
                f"after={update_result.after_history_end}",
                f"appended={update_result.appended_rows}",
                f"reason={update_result.reason}",
            )
    if mode in {"scoreopt_t1430_dispersion", "scoreopt_t1430_fusion_gate"}:
        state_feature_path = str(switch_cfg.get("state_feature_path", "")).strip()
        if not state_feature_path:
            raise ValueError(f"{mode} requires model_switch.state_feature_path")
        update_t1430_market_state_feature_history(
            state_feature_path=state_feature_path,
            clean_root=clean_root,
            score_day=score_day,
            price_field=str(switch_cfg.get("state_price_field", "last")).strip() or "last",
        )
        decision = (
            decide_scoreopt_t1430_fusion_gate(switch_cfg, score_day=score_day)
            if mode == "scoreopt_t1430_fusion_gate"
            else decide_scoreopt_t1430_dispersion(switch_cfg, score_day=score_day)
        )
    elif mode == "scoreopt_bm_short":
        decision = decide_scoreopt_bm_short(switch_cfg, score_day=score_day)
    elif mode == "regime_bm20_sign":
        decision = decide_single_challenger_by_regime(switch_cfg, score_day=score_day)
    else:
        decision = decide_single_challenger_by_sharpe(switch_cfg, score_day=score_day)
    _assert_no_live_model_switch_fallback(decision, switch_cfg=switch_cfg)
    stale_policy = str(switch_cfg.get("stale_history_policy", "champion")).strip().lower()
    if expected_history_end is not None and decision.history_end != expected_history_end:
        reason = f"stale_return_history_expected_{expected_history_end:%Y-%m-%d}"
        if stale_policy in {"champion", "fallback_champion", "fallback_to_champion"}:
            decision = replace(
                decision,
                selected_model_id=decision.champion_model_id,
                selected_name=decision.champion_name,
                reason=reason,
            )
        elif stale_policy == "fail":
            raise RuntimeError(
                "model_switch return history is stale: "
                f"expected_history_end={expected_history_end}, actual={decision.history_end}"
            )
        else:
            raise ValueError(f"unsupported model_switch.stale_history_policy: {stale_policy}")
    model_switch_warnings = _model_switch_warnings(decision)
    _append_live_day_notes(
        paths_cfg=paths_cfg,
        day=score_day,
        warnings=model_switch_warnings,
    )
    selected_score_df = current_score_df
    challenger_by_model = {str(item["model_id"]): item for item in challenger_results}
    selected_challenger = challenger_by_model.get(decision.selected_model_id)
    if selected_challenger is not None:
        selected_score_df = selected_challenger["score_df"]
    elif decision.selected_model_id != current_model_id:
        raise ValueError(f"model_switch selected unknown model_id: {decision.selected_model_id}")

    extra = {
        "challenger_score_path": (
            str(selected_challenger["score_path"])
            if selected_challenger is not None
            else str(challenger_results[0]["score_path"])
        ),
        "challenger_score_paths": {
            str(item["model_id"]): str(item["score_path"])
            for item in challenger_results
        },
        "source_scores": [
            {
                "name": str(item["name"]),
                "model_id": str(item["model_id"]),
                "score_path": str(item["score_path"]),
                "rows": int(len(item["score_df"])),
                "source_scores": item["source_scores"],
            }
            for item in challenger_results
        ],
        "shadow_return_updates": [item.to_dict() for item in shadow_update_results],
        "warnings": model_switch_warnings,
    }
    print(
        "live model switch decision:",
        f"selected={decision.selected_name}",
        f"mode={decision.mode}",
        f"metric={decision.metric}",
        f"champion_score={decision.champion_score}",
        f"challenger_score={decision.challenger_score}",
        f"diff={decision.score_diff}",
        f"history_end={decision.history_end}",
        f"reason={decision.reason}",
    )
    return decision.selected_model_id, selected_score_df, decision, extra


def _backfill_live_history(
    *,
    paths_cfg: dict,
    raw_root: str,
    score_day: date,
    prev_trade_day: date,
    panel_cfg: dict,
    label_cfg: dict,
    factor_cfg: dict,
    panel_name: str,
    window_days: int,
    skip_panel_build: bool = False,
) -> None:
    history_days = prev_trading_days_from_raw(
        raw_root,
        score_day,
        max(1, int(window_days)),
        kind="snapshot",
        asset="cbond",
    )
    if not history_days:
        history_days = [prev_trade_day]
    history_days = sorted(set(history_days))

    assets = _normalize_assets(panel_cfg.get("assets", []))
    if not assets:
        assets = ["cbond"]

    missing_panel_by_asset: dict[str, list[date]] = {}
    if skip_panel_build:
        print(
            "live backfill panel skipped:",
            "reason=clean_direct_factor_panel_source",
            f"history_days={len(history_days)}",
        )
    else:
        for asset in assets:
            missing_panel_days = _missing_days(
                history_days,
                path_builder=lambda d, _asset=asset: _panel_day_path(paths_cfg, d, asset=_asset, panel_name=panel_name),
            )
            if missing_panel_days:
                missing_panel_by_asset[asset] = missing_panel_days

    if (not skip_panel_build) and missing_panel_by_asset:
        all_missing_panel_days = sorted(
            {
                day
                for asset_days in missing_panel_by_asset.values()
                for day in asset_days
            }
        )
        panel_start, panel_end = _build_day_span(all_missing_panel_days)
        panel_backfill_cfg = dict(panel_cfg)
        panel_backfill_cfg["start"] = panel_start
        panel_backfill_cfg["end"] = panel_end
        panel_backfill_cfg["refresh"] = False
        panel_backfill_cfg["overwrite"] = False
        print(
            "live backfill panel:",
            f"start={panel_start}",
            f"end={panel_end}",
            f"assets={','.join(sorted(missing_panel_by_asset.keys()))}",
            f"missing_days={len(all_missing_panel_days)}",
        )
        panel_result = run_panel_build(
            start=panel_start,
            end=panel_end,
            refresh=False,
            overwrite=False,
            cfg=panel_backfill_cfg,
        )
        print("live backfill panel done:", panel_result)
        for asset, asset_days in missing_panel_by_asset.items():
            for day in asset_days:
                _require_existing(
                    _panel_day_path(paths_cfg, day, asset=asset, panel_name=panel_name),
                    name=f"{asset} panel(backfill)",
                )

    missing_label_days = _missing_days(
        history_days,
        path_builder=lambda d: _label_day_path(paths_cfg, d),
    )
    if missing_label_days:
        label_start, _ = _build_day_span(missing_label_days)
        label_end = score_day
        label_backfill_cfg = dict(label_cfg)
        label_backfill_cfg["start"] = label_start
        label_backfill_cfg["end"] = label_end
        label_backfill_cfg["refresh"] = False
        label_backfill_cfg["overwrite"] = False
        print(
            "live backfill labels:",
            f"start={label_start}",
            f"end={label_end}",
            f"missing_days={len(missing_label_days)}",
        )
        label_result = run_label_build(
            start=label_start,
            end=label_end,
            refresh=False,
            overwrite=False,
            cfg=label_backfill_cfg,
            panel_cfg=panel_cfg,
        )
        print("live backfill labels done:", label_result)
        for day in missing_label_days:
            _require_existing(_label_day_path(paths_cfg, day), name="label(backfill)")

    missing_factor_days = _missing_days(
        history_days,
        path_builder=lambda d: _factor_day_path(paths_cfg, d, panel_name=panel_name),
    )
    if missing_factor_days:
        factor_start, factor_end = _build_day_span(missing_factor_days)
        factor_backfill_cfg = dict(factor_cfg)
        factor_backfill_cfg["start"] = factor_start
        factor_backfill_cfg["end"] = factor_end
        factor_backfill_cfg["refresh"] = False
        factor_backfill_cfg["overwrite"] = False
        factor_backfill_cfg["panel_name"] = panel_name
        print(
            "live backfill factors:",
            f"start={factor_start}",
            f"end={factor_end}",
            f"missing_days={len(missing_factor_days)}",
        )
        factor_result = run_factor_build(
            start=factor_start,
            end=factor_end,
            refresh=False,
            overwrite=False,
            cfg=factor_backfill_cfg,
        )
        print("live backfill factors done:", factor_result)
        for day in missing_factor_days:
            _require_existing(_factor_day_path(paths_cfg, day, panel_name=panel_name), name="factor(backfill)")


def _allowlist_diagnostics(pool_info: dict, pre_count: int, post_count: int) -> dict:
    return {
        "allowlist_table": pool_info.get("allowlist_table") or pool_info.get("pool_table"),
        "allowlist_lag_trading_days": pool_info.get("allowlist_lag_trading_days")
        or pool_info.get("pool_lag_trading_days"),
        "allowlist_day_expected": pool_info.get("allowlist_day_expected") or pool_info.get("pool_day_expected"),
        "allowlist_day_used": pool_info.get("allowlist_day_used") or pool_info.get("pool_day_used"),
        "allowlist_applied": bool(pool_info.get("allowlist_applied") or pool_info.get("pool_enabled")),
        "allowlist_codes_count": int(pool_info.get("allowlist_codes_count") or pool_info.get("pool_codes_count") or 0),
        "allowlist_fallback_no_filter": bool(
            pool_info.get("allowlist_fallback_no_filter") or pool_info.get("fallback_no_filter")
        ),
        "allowlist_fallback_reason": pool_info.get("allowlist_fallback_reason") or pool_info.get("fallback_reason", ""),
        "pre_allowlist_count": int(pre_count),
        "post_allowlist_count": int(post_count),
    }


def _assert_live_data_boundary(live_cfg: dict) -> None:
    """Live consumes DataHub clean data, then builds CBOND_ON local derived artifacts."""

    forbidden_top_level = {"source", "redis"}
    present_top_level = sorted(k for k in forbidden_top_level if k in live_cfg)
    if present_top_level:
        raise ValueError(
            "live_config must not own raw/redis processing; remove sections: "
            + ", ".join(present_top_level)
        )

    data_cfg = dict(live_cfg.get("data", {}))
    forbidden_data_keys = {
        "refresh",
        "overwrite",
        "lookback_days",
        "kline_enabled",
        "snapshot_source",
        "raw_sync_mode_when_redis",
        "redis_sync_day",
        "redis_source",
        "redis_stage",
        "redis_asset_type",
        "redis_incremental",
        "redis_full_day",
        "min_price",
        "max_price",
        "min_amount",
        "min_volume",
    }
    present_data_keys = sorted(k for k in forbidden_data_keys if k in data_cfg)
    if present_data_keys:
        raise ValueError(
            "live_config.data must not own raw/redis processing; remove keys: "
            + ", ".join(present_data_keys)
        )

    if "security_banlist" in live_cfg:
        raise ValueError("live universe filter must only use o_0005 allowlist; remove security_banlist")

    allowlist_raw = live_cfg.get("allowlist")
    if not isinstance(allowlist_raw, dict):
        raise ValueError("live allowlist must be configured as an o_0005 allowlist block")
    if not bool(allowlist_raw.get("enabled", True)):
        raise ValueError("live allowlist must be enabled; o_0005 is the only allowed universe filter")
    table = str(allowlist_raw.get("table", allowlist_raw.get("pool_table", ""))).strip()
    if table != O005_ALLOWLIST_TABLE:
        raise ValueError(
            "live allowlist table must be "
            f"{O005_ALLOWLIST_TABLE}; got {table or '<missing>'}"
        )


def _prev_trading_day(raw_root: str, day: date) -> date:
    prev_days = prev_trading_days_from_raw(
        raw_root,
        day,
        1,
        kind="snapshot",
        asset="cbond",
    )
    if not prev_days:
        raise RuntimeError(f"cannot resolve previous trading day before {day}")
    return prev_days[-1]


def run_once(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> Path:
    _ = mode
    paths_cfg = load_config_file("paths")
    live_cfg = load_config_file("live")
    _assert_live_data_boundary(live_cfg)

    schedule_cfg = dict(live_cfg.get("schedule", {}))
    model_cfg = dict(live_cfg.get("model_score", {}))
    strategy_cfg = dict(live_cfg.get("strategy", {}))
    output_cfg = dict(live_cfg.get("output", {}))
    allowlist_raw = live_cfg.get("allowlist", {})
    allowlist_cfg = allowlist_raw if isinstance(allowlist_raw, dict) else {}
    allowlist_enabled = bool(allowlist_cfg.get("enabled", True)) if isinstance(allowlist_raw, dict) else bool(allowlist_raw)
    factor_cfg_key, live_factor_cfg = load_live_factor_runtime(live_cfg)
    model_cfg_key, live_model_score_cfg, model_id = load_live_model_runtime(live_cfg)

    assert_no_date_fields_in_live_config(schedule_cfg, model_cfg)

    today = today_shanghai()
    target_day = parse_date(target) if target is not None else today
    score_day = parse_date(start) if start is not None else (
        today if target_day >= today else target_day
    )

    raw_root = str(paths_cfg["raw_data_root"])
    clean_root = str(paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root"))
    data_hub = data_hub_runtime_from_live(
        live_cfg,
        raw_root=raw_root,
        clean_root=clean_root,
    )

    prev_trade_day = _prev_trading_day(raw_root, score_day)

    print(
        "live run window:",
        f"score_day={score_day}",
        f"target_day={target_day}",
        f"prev_trading_day={prev_trade_day}",
        "mode=clean_consumer_build_local",
    )

    if not bool(data_hub.get("ready_gate_enabled", True)):
        raise ValueError("live_config.data_hub.ready_gate_enabled must be true in consumer-only mode")
    ensure_publish_ready(
        runtime=data_hub,
        trade_day=score_day,
    )

    panel_cfg = dict(load_config_file("panel"))
    panel_cfg["start"] = score_day
    panel_cfg["end"] = score_day
    panel_cfg["refresh"] = True
    panel_cfg["overwrite"] = True
    panel_name = str(panel_cfg.get("panel_name") or live_factor_cfg.get("panel_name") or "T1430")
    panel_cfg["panel_name"] = panel_name

    label_cfg = dict(load_config_file("label"))
    label_cfg["start"] = prev_trade_day
    label_cfg["end"] = score_day
    label_cfg["refresh"] = True
    label_cfg["overwrite"] = True

    factor_runtime_cfg = dict(live_factor_cfg)
    factor_runtime_cfg["start"] = score_day
    factor_runtime_cfg["end"] = score_day
    factor_runtime_cfg["refresh"] = True
    factor_runtime_cfg["overwrite"] = True
    factor_runtime_cfg["panel_name"] = panel_name
    panel_source_mode = _factor_panel_source_mode(factor_runtime_cfg)
    use_clean_direct_panel_source = panel_source_mode == "clean_direct"

    window_days = _parse_live_model_window_days(live_model_score_cfg, model_id)
    if window_days > 0:
        print(
            "live backfill check:",
            f"window_days={window_days}",
            f"score_day={score_day}",
        )
        _backfill_live_history(
            paths_cfg=paths_cfg,
            raw_root=raw_root,
            score_day=score_day,
            prev_trade_day=prev_trade_day,
            panel_cfg=panel_cfg,
            label_cfg=label_cfg,
            factor_cfg=factor_runtime_cfg,
            panel_name=panel_name,
            window_days=window_days,
            skip_panel_build=use_clean_direct_panel_source,
        )

    if use_clean_direct_panel_source:
        print(
            "live build panel skipped:",
            f"day={score_day}",
            f"panel={panel_name}",
            f"panel_source={panel_source_mode}",
        )
    else:
        print("live build panel:", f"day={score_day}", f"panel={panel_name}")
        panel_result = run_panel_build(
            start=score_day,
            end=score_day,
            refresh=True,
            overwrite=True,
            cfg=panel_cfg,
        )
        _require_existing(_panel_day_path(paths_cfg, score_day, asset="cbond", panel_name=panel_name), name="cbond panel")
        if "stock" in _normalize_assets(panel_cfg.get("assets", [])):
            _require_existing(_panel_day_path(paths_cfg, score_day, asset="stock", panel_name=panel_name), name="stock panel")
        print("live build panel done:", panel_result)

    print("live build labels:", f"day={prev_trade_day}", f"next_day={score_day}")
    label_result = run_label_build(
        start=prev_trade_day,
        end=score_day,
        refresh=True,
        overwrite=True,
        cfg=label_cfg,
        panel_cfg=panel_cfg,
    )
    _require_existing(_label_day_path(paths_cfg, prev_trade_day), name="label")
    print("live build labels done:", label_result)

    print("live build factors:", f"day={score_day}", f"panel={panel_name}")
    factor_result = run_factor_build(
        start=score_day,
        end=score_day,
        refresh=True,
        overwrite=True,
        cfg=factor_runtime_cfg,
    )
    _require_existing(_factor_day_path(paths_cfg, score_day, panel_name=panel_name), name="factor")
    print("live build factors done:", factor_result)

    model_start = score_day
    model_end = score_day
    model_label_cutoff = model_cfg.get("label_cutoff")
    if model_label_cutoff is None:
        model_label_cutoff = prev_trade_day
    model_label_cutoff = parse_date(model_label_cutoff)
    print(
        "model score window:",
        f"score_day={score_day}",
        f"label_cutoff={model_label_cutoff}",
    )
    print(
        "live config profile:",
        f"factors={factor_cfg_key}",
        f"models={model_cfg_key}",
        f"model_id={model_id}",
    )

    model_result = run_model_score(
        model_id=model_id,
        start=model_start,
        end=model_end,
        label_cutoff=model_label_cutoff,
        cfg=live_model_score_cfg,
    )
    score_path = _resolve_model_result_score_path(model_result, model_id=model_id, paths_cfg=paths_cfg)
    score_df = _score_df_from_path(score_path, score_day)

    out_dir = Path(paths_cfg["results_root"]) / "live" / f"{target_day:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy_id = str(strategy_cfg.get("strategy_id", "strategy01_topk_turnover"))
    strategy_config = load_strategy_config(strategy_cfg.get("strategy_config_path"))
    strategy_config = strategy_config or {k: v for k, v in strategy_cfg.items() if k != "strategy_id"}
    selected_model_id = model_id
    switch_decision: SwitchDecision | None = None
    switch_extra: dict = {}
    switch_raw = live_cfg.get("model_switch", {})
    switch_cfg = switch_raw if isinstance(switch_raw, dict) else {}
    if bool(switch_cfg.get("enabled", False)):
        selected_model_id, score_df, switch_decision, switch_extra = _apply_live_model_switch(
            switch_cfg=switch_cfg,
            current_model_id=model_id,
            current_score_path=score_path,
            current_score_df=score_df,
            score_day=score_day,
            expected_history_end=prev_trade_day,
            label_cutoff=model_label_cutoff,
            raw_root=raw_root,
            clean_root=clean_root,
            strategy_id=strategy_id,
            strategy_config=strategy_config,
            allowlist_cfg=allowlist_cfg,
            paths_cfg=paths_cfg,
        )
        write_switch_decision(
            out_dir / "model_switch_decision.json",
            switch_decision,
            extra=switch_extra,
        )

    clean_daily = read_clean_daily(clean_root, score_day)
    if clean_daily.empty:
        raise RuntimeError(
            f"clean daily data missing for {score_day}; live will not fall back to score-only universe"
        )
    universe = clean_daily.merge(score_df[["code", "score"]], on="code", how="inner")
    if universe.empty:
        raise ValueError("no score matched to clean data")

    pool_cfg = load_upstream_pool_config(allowlist_cfg or None)
    pool_codes, pool_info = resolve_pool_codes_for_trade_day(
        raw_data_root=raw_root,
        trade_day=score_day,
        pool_cfg=pool_cfg,
        enabled=allowlist_enabled,
    )
    if bool(pool_info.get("fallback_no_filter", False)):
        raise RuntimeError(
            "[allowlist] required pool is unavailable; live does not allow no-filter fallback: "
            f"trade_day={score_day:%Y-%m-%d} "
            f"expected_pool_day={pool_info.get('pool_day_expected')} "
            f"reason={pool_info.get('fallback_reason')} "
            f"nearest_pool_day={pool_info.get('nearest_pool_day')}"
        )
    pre_allowlist_count = int(len(universe))
    universe = apply_allowlist_filter_to_universe(universe, allowlist_codes=pool_codes)
    allowlist_diag = _allowlist_diagnostics(pool_info, pre_allowlist_count, int(len(universe)))
    if universe.empty:
        raise ValueError("live universe is empty after allowlist filter")

    prev_positions = load_previous_holdings(Path(paths_cfg["results_root"]) / "live", target_day)
    picks = select_signals(
        SignalSelectionRequest(
            universe=universe[["code", "score"]],
            trade_date=target_day,
            prev_positions=prev_positions,
            strategy_id=strategy_id,
            strategy_config=strategy_config,
        )
    )
    if picks.empty:
        raise ValueError("strategy returned empty picks")

    picks = picks.copy()
    picks["signal_day"] = score_day
    picks["buy_day"] = score_day
    picks["sell_day"] = target_day
    picks["target_day"] = target_day
    picks["score_day"] = score_day
    picks["trade_date"] = target_day
    picks.to_csv(out_dir / "trade_list.csv", index=False)
    allowlist_summary = {
        **allowlist_diag,
        "universe_filter": "o_0005_only",
        "target_day": target_day,
        "score_day": score_day,
        "picks_count": int(len(picks)),
        "base_model_id": model_id,
        "selected_model_id": selected_model_id,
        "model_switch_enabled": bool(switch_cfg.get("enabled", False)),
    }
    if switch_decision is not None:
        allowlist_summary["model_switch_reason"] = switch_decision.reason
        allowlist_summary["model_switch_history_end"] = switch_decision.history_end
    summary_text = json.dumps(allowlist_summary, ensure_ascii=False, indent=2, default=str)
    (out_dir / "allowlist_summary.json").write_text(summary_text, encoding="utf-8")
    (out_dir / "universe_filter_summary.json").write_text(summary_text, encoding="utf-8")

    if bool(output_cfg.get("db_write", False)):
        if not output_cfg.get("db_table"):
            raise ValueError("live_config.output.db_table is required when db_write=true")
        db_trade_day = prev_trade_day
        db_picks = picks.copy()
        db_picks["trade_date"] = db_trade_day
        try:
            write_trades_to_db(
                trades=db_picks,
                trade_day=db_trade_day,
                table=str(output_cfg["db_table"]),
                mode=str(output_cfg.get("db_mode", "replace_date")),
                backend=output_cfg.get("db_backend"),
            )
        except FileNotFoundError as exc:
            print(f"skip output db write: {exc}")

    return out_dir


def run(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> Path:
    return run_once(start=start, target=target, mode=mode)

